'''
Function:
    define the weekly supervised deep detection network
Author:
    Charles
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.losses import *
from modules.backbones import *
from libs.roi_pool.roi_pool import roi_pool
from libs.roi_align.roi_align import roi_align


'''wsdnn base'''
class WSDNNBase(nn.Module):
    def __init__(self, cfg, mode, logger_handle, **kwargs):
        super(WSDNNBase, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger_handle = logger_handle
        # backbone network
        self.base_model = None
        # top model
        self.top_model = None
        # cls branch
        self.fc_cls = None
        # det branch
        self.fc_det = None
    '''forward'''
    def forward(self, x, proposals, proposals_scores, targets=None):
        batch_size = x.size(0)
        # obtain features
        x = self.base_model(x)
        # roi pool/align
        rois = torch.zeros(batch_size, proposals.size(1), 5).type_as(proposals)
        for i in range(batch_size):
            rois[i, :, 0] = i
            rois[i, :, 1:] = proposals[i]
        if self.cfg.BACKBONE_CFG['rois']['type'] == 'roi_pool':
            pooled_features = roi_pool(x, rois.view(-1, 5), self.cfg.BACKBONE_CFG['rois']['size'], 1.0/self.stride)
        elif self.cfg.BACKBONE_CFG['rois']['type'] == 'roi_align':
            pooled_features = roi_align(x, rois.view(-1, 5), self.cfg.BACKBONE_CFG['rois']['size'], 1.0/self.stride)
        else:
            raise ValueError('Unsupport rois.type %s...' % self.cfg.BACKBONE_CFG['rois']['type'])
        # apply proposal scores
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        pooled_features = pooled_features * proposals_scores.view(pooled_features.size(0), 1)
        pooled_features = self.top_model(pooled_features)
        # classification
        x_cls = F.softmax(self.fc_cls(pooled_features), dim=1)
        # detection
        x_det = self.fc_det(pooled_features)
        x_det = x_det.view(batch_size, proposals.size(1), -1)
        x_det = F.softmax(x_det, dim=1)
        x_det = x_det.view(pooled_features.size(0), -1)
        # combine
        preds_cls = x_cls * x_det
        preds_cls = preds_cls.view(batch_size, proposals.size(1), -1)
        # calculate loss
        loss_cls = torch.Tensor([0]).type_as(pooled_features)
        if self.mode == 'TRAIN':
            image_level_scores = preds_cls.sum(1)
            image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
            if self.cfg.LOSS_CFG['cls_loss']['type'] == 'binary_ce':
                loss_cls = BinaryCrossEntropyLoss(preds=image_level_scores,
                                                  targets=targets, 
                                                  loss_weight=self.cfg.LOSS_CFG['cls_loss']['binary_ce']['weight'], 
                                                  size_average=self.cfg.LOSS_CFG['cls_loss']['binary_ce']['size_average'], 
                                                  avg_factor=batch_size)
            else:
                raise ValueError('Unsupport classification loss type %s...' % self.cfg.LOSS_CFG['cls_loss']['type'])
        # return necessary infos
        return preds_cls, loss_cls
    '''set bn fixed'''
    @staticmethod
    def setBnFixed(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters():
                p.requires_grad = False
    '''set bn eval'''
    @staticmethod
    def setBnEval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()


'''wsdnn using VGG as backbone'''
class WSDNNVGG(WSDNNBase):
    stride = 16
    def __init__(self, cfg, mode, logger_handle, **kwargs):
        WSDNNBase.__init__(self, cfg, mode, logger_handle)
        # backbone network
        if self.mode == 'TRAIN':
            if cfg.BACKBONE_CFG['pretrained_model_path']:
                self.backbone = VGGs(cfg.BACKBONE_CFG['type'])
                self.backbone.load_state_dict(torch.load(cfg.BACKBONE_CFG['pretrained_model_path']))
                self.logger_handle.info('Loading pretrained weights from %s for backbone network...' % self.pretrained_model_path)
            else:
                self.backbone = VGGs(cfg.BACKBONE_CFG['type'], True)
        self.base_model = self.backbone.features[:-1]
        # top model
        self.top_model = self.backbone.classifier[:-1]
        in_features = 4096
        # cls branch
        self.fc_cls = nn.Linear(in_features=in_features, out_features=cfg.DATASET_CFG['num_classes'])
        # det branch
        self.fc_det = nn.Linear(in_features=in_features, out_features=cfg.DATASET_CFG['num_classes'])
    '''set train mode'''
    def setTrain(self):
        nn.Module.train(self, True)
        self.base_model.apply(WSDNNBase.setBnEval)
        self.top_model.apply(WSDNNBase.setBnEval)
        self.fc_cls.apply(WSDNNBase.setBnEval)
        self.fc_det.apply(WSDNNBase.setBnEval)