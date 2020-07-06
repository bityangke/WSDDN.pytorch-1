'''
Function:
    load voc dataset
Author:
    Charles
'''
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from scipy.io import loadmat
from PIL import Image, ImageOps
from modules.utils.misc import *
from torch.utils.data import Dataset
from torchvision.transforms import transforms


'''voc dataset'''
class VOCDataset(Dataset):
    clsname2id = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }
    def __init__(self, dataset_cfg, proposals_cfg, mode, **kwargs):
        self.dataset_cfg = dataset_cfg
        self.proposals_cfg = proposals_cfg
        self.imageids, self.proposals, self.proposals_scores = self.parseMATFile(proposals_cfg['proposals_filepath'])
        self.mode = mode
        assert self.mode in ['TRAIN', 'TEST']
        if dataset_cfg['type'] == 'VOC07':
            root_dir = os.path.join(dataset_cfg['root_dir'], 'VOCdevkit/VOC2007/')
            self.imagepaths = [os.path.join(root_dir, 'JPEGImages', f'{imageid}.jpg') for imageid in self.imageids]
            self.annotations_paths = [os.path.join(root_dir, 'Annotations', f'{imageid}.xml') for imageid in self.imageids]
        else:
            raise ValueError('Unsupport dataset type %s for VOCDataset...' % dataset_cfg['type'])
        if mode == 'TRAIN':
            self.image_ratios, self.imageids, self.imagepaths, self.annotations_paths, self.proposals, self.proposals_scores = \
                    self.filterImages(self.imageids, self.imagepaths, self.annotations_paths, self.proposals, self.proposals_scores)
        self.max_image_size = None
        self.num_iters = 0
    '''get item'''
    def __getitem__(self, index):
        # proposals: (x_min, y_min, x_max, y_max)
        proposals = self.proposals[index].astype(np.float32)
        proposals = np.stack((proposals[:, 1], proposals[:, 0], proposals[:, 3], proposals[:, 2]), axis=1)
        # filter small proposals
        mask = filterSmallBoxes(proposals, 20)
        proposals = proposals[mask]
        # proposal scores: (num_proposals, 1)
        proposals_scores = self.proposals_scores[index].astype(np.float32)
        proposals_scores = proposals_scores[mask]
        # select topk
        topk = np.argsort(proposals_scores.reshape(-1), axis=0)[-self.proposals_cfg['num_proposals']:]
        while topk.shape[0] < self.proposals_cfg['num_proposals']:
            random_index = np.random.permutation(np.arange(topk.shape[0]))[:self.proposals_cfg['num_proposals']-topk.shape[0]]
            topk = np.concatenate([topk, topk[random_index]], axis=0)
        topk = np.random.permutation(topk)
        proposals = proposals[topk]
        proposals_scores = proposals_scores[topk]
        # get ground truths
        xml = ET.parse(self.annotations_paths[index])
        width = int(xml.find('size').find('width').text)
        height = int(xml.find('size').find('height').text)
        gt_boxes, gt_labels = [], []
        for obj in xml.findall('object'):
            if obj.find('difficult').text != '1':
                bndbox = obj.find('bndbox')
                gt_box = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                gt_box[0] -= 1
                gt_box[1] -= 1
                if (gt_box[2] <= gt_box[0]) or (gt_box[3] <= gt_box[1]):
                    continue
                gt_boxes.append(gt_box)
                gt_labels.append(self.clsname2id[obj.find('name').text])
        gt_boxes = np.stack(gt_boxes).astype(np.float32)
        gt_labels = np.stack(gt_labels).astype(np.int32)
        # training
        if self.mode == 'TRAIN':
            self.num_iters += 1
            # --preprocess image
            image = Image.open(self.imagepaths[index]).convert('RGB')
            assert image.width == width and image.height == height, 'something error when reading image or annotation'
            if random.random() > 0.5:
                image = ImageOps.mirror(image)
                gt_boxes[:, [0, 2]] = image.width - gt_boxes[:, [2, 0]] - 1
                proposals[:, [0, 2]] = image.width - proposals[:, [2, 0]] - 1
            if (self.max_image_size is None) or (self.num_iters % self.dataset_cfg['batch_size'] == 0):
                self.max_image_size = random.choice(self.dataset_cfg['scales'])
            image, scale_factor, target_size = VOCDataset.preprocess(image=image,
                                                                     style=self.dataset_cfg['style'],
                                                                     max_size=self.max_image_size,
                                                                     use_color_jitter=self.dataset_cfg['use_color_jitter'])
            # --correct gt_boxes and proposals
            proposals = proposals * scale_factor
            gt_boxes = gt_boxes * scale_factor
            # --padding
            gt_boxes_padding = np.zeros((self.dataset_cfg['max_num_gt_boxes'], 4), dtype=np.float32)
            gt_boxes_padding[range(len(gt_boxes))[:self.dataset_cfg['max_num_gt_boxes']]] = gt_boxes[:self.dataset_cfg['max_num_gt_boxes']]
            # convert to one-hot
            gt_labels_one_hot = np.full(20, 0, dtype=np.float32)
            for label in gt_labels:
                gt_labels_one_hot[label] = 1.0
            # --return the necessary data
            return int(self.imageids[index]), image, torch.from_numpy(proposals), torch.from_numpy(proposals_scores), torch.from_numpy(gt_labels_one_hot)
        # testing
        else:
            # --preprocess image (multi-scale testing by default for wsod)
            image_ori = Image.open(self.imagepaths[index]).convert('RGB')
            assert image_ori.width == width and image_ori.height == height, 'something error when reading image or annotation'
            proposals_ori = proposals.copy()
            images_list = []
            proposals_list = []
            scale_factors_list = []
            proposals_scores_list = []
            for is_flip in [False, True]:
                for max_size in self.dataset_cfg['scales']:
                    image, scale_factor, target_size = VOCDataset.preprocess(image=image,
                                                                             style=self.dataset_cfg['style'],
                                                                             max_size=max_size,
                                                                             use_color_jitter=False)
                    images_list.append(image)
                    proposals = proposals_ori * scale_factor
                    if is_flip:
                        proposals[:, [0, 2]] = image.width - proposals[:, [2, 0]] - 1
                    proposals_list.append(proposals)
                    scale_factors_list.append(scale_factor)
                    proposals_scores_list.append(proposals_scores)
            # --return the necessary data
            return int(self.imageids[index]), images_list, proposals_list, proposals_scores_list, gt_boxes, gt_labels
    '''parse mat file'''
    def parseMATFile(self, filepath):
        data = loadmat(filepath)
        proposals = data['boxes'][0]
        proposals_scores = data['boxScores'][0]
        imageids = [str(item[0]) for item in data['images'][0]]
        return imageids, proposals, proposals_scores
    '''filter images'''
    def filterImages(self, imageids, imagepaths, annotations_paths, proposals, proposals_scores):
        image_ratios = []
        imageids_filtered = []
        imagepaths_filtered = []
        annotations_paths_filtered = []
        proposals_filtered = []
        proposals_scores_filtered = []
        for imageid, imagepath, annpath, proposals_each, proposals_scores_each in zip(imageids, imagepaths, annotations_paths, proposals, proposals_scores):
            xml = ET.parse(annpath)
            width = int(xml.find('size').find('width').text)
            height = int(xml.find('size').find('height').text)
            gt_boxes = []
            for obj in xml.findall('object'):
                if obj.find('difficult').text != '1':
                    bndbox = obj.find('bndbox')
                    gt_box = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                    gt_box[0] -= 1
                    gt_box[1] -= 1
                    if (gt_box[2] <= gt_box[0]) or (gt_box[3] <= gt_box[1]):
                        continue
                    gt_boxes.append(gt_box)
            if len(gt_boxes) > 0:
                image_ratios.append(float(width/height))
                imageids_filtered.append(imageid)
                imagepaths_filtered.append(imagepath)
                annotations_paths_filtered.append(annpath)
                proposals_filtered.append(proposals_each)
                proposals_scores_filtered.append(proposals_scores_each)
        return image_ratios, imageids_filtered, imagepaths_filtered, annotations_paths_filtered, proposals_filtered, proposals_scores_filtered
    '''length'''
    def __len__(self):
        return len(self.imageids)
    '''preprocess image'''
    @staticmethod
    def preprocess(image, style, max_size, use_color_jitter=False):
        assert style in ['caffe', 'pytorch']
        w_ori, h_ori = image.width, image.height
        if w_ori > h_ori:
            scale_factor = max_size / w_ori
        else:
            scale_factor = max_size / h_ori
        target_size = (round(scale_factor*h_ori), round(scale_factor*w_ori))
        if style == 'caffe':
            means_norm = (0.4814576470588235, 0.4546921568627451, 0.40384352941176466)
            stds_norm = (1., 1., 1.)
        else:
            means_norm = (0.485, 0.456, 0.406)
            stds_norm = (0.229, 0.224, 0.225)
        if use_color_jitter:
            transform = transforms.Compose([transforms.Resize(target_size),
                                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=means_norm, std=stds_norm)])
        else:
            transform = transforms.Compose([transforms.Resize(target_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=means_norm, std=stds_norm)])
        image = transform(image)
        if style == 'caffe': image = image[(2, 1, 0), :, :] * 255
        return image, scale_factor, target_size
    '''padding collate fn'''
    @staticmethod
    def paddingCollateFn(data_batch):
        # data_batch: [[image_id, image(channel, height, width), proposals, proposals_scores, gt_labels], ...]
        max_height = max([data[1].shape[1] for data in data_batch])
        max_width = max([data[1].shape[2] for data in data_batch])
        # get new data_batch
        imageid_batch = []
        image_batch = []
        proposals_batch = []
        proposals_scores_batch = []
        gt_labels_batch = []
        for data in data_batch:
            image_id, image, proposals, proposals_scores, gt_labels = data
            # (left, right, top, bottom)
            image_padding = F.pad(input=image, pad=(0, max_width-image.shape[2], 0, max_height-image.shape[1]))
            imageid_batch.append(torch.from_numpy(np.array([image_id])))
            image_batch.append(image_padding)
            proposals_batch.append(proposals)
            proposals_scores_batch.append(proposals_scores)
            gt_labels_batch.append(gt_labels)
        imageid_batch = torch.stack(imageid_batch, dim=0)
        image_batch = torch.stack(image_batch, dim=0)
        proposals_batch = torch.stack(proposals_batch, dim=0)
        proposals_scores_batch = torch.stack(proposals_scores_batch, dim=0)
        gt_labels_batch = torch.stack(gt_labels_batch, dim=0)
        return imageid_batch, image_batch, proposals_batch, proposals_scores_batch, gt_labels_batch
    '''evaluate mAP'''
    @staticmethod
    def evaluate():
        pass