'''
Function:
    train the model
Author:
    Charles
'''
import os
import torch
import warnings
import argparse
import torch.nn as nn
from modules.utils import *
from modules.optimizer import *
from modules.WSDDN import WSDNNVGG
from cfgs.getcfg import getCfgByDatasetAndBackbone
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='Weekly supervised deep detection network')
    parser.add_argument('--datasetname', dest='datasetname', help='dataset for training.', default='', type=str, required=True)
    parser.add_argument('--backbonename', dest='backbonename', help='backbone network for training.', default='', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str)
    args = parser.parse_args()
    return args


'''train the model'''
def train():
    # prepare base things
    args = parseArgs()
    cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
    record_cfg = cfg.RECORD_CFG['train']
    checkDir(record_cfg['backupdir'])
    logger_handle = Logger(record_cfg['logfile'])
    use_cuda = torch.cuda.is_available()
    is_multi_gpus = cfg.BACKBONE_CFG['is_multi_gpus']
    if is_multi_gpus: assert use_cuda
    # prepare dataset
    if args.datasetname.upper() == 'VOC07':
        proposals_filepath = os.path.join(cfg.PROPOSALS_CFG['train']['root_dir'], cfg.PROPOSALS_CFG['train']['filename'])
        proposals_cfg = {'proposals_filepath': proposals_filepath, 'num_proposals': cfg.PROPOSALS_CFG['train']['num_proposals']}
        dataset = VOCDataset(cfg.DATASET_CFG, proposals_cfg, 'TRAIN')
        dataloader_cfg = {'batch_size': cfg.DATASET_CFG['batch_size'],
                          'pin_memory': cfg.DATASET_CFG['pin_memory'],
                          'num_workers': cfg.DATASET_CFG['num_workers'],
                          'sampler': GroupSampler,
                          'collate_fn': VOCDataset.paddingCollateFn}
        dataloader = buildDataloader(dataset, cfg=dataloader_cfg, mode='TRAIN')
    else:
        raise ValueError('Unsupport datasetname %s now...' % args.datasetname)
    # prepare model
    if args.backbonename.find('vgg') != -1:
        model = WSDNNVGG(cfg=cfg, mode='TRAIN', logger_handle=logger_handle)
    else:
        raise ValueError('Unsupport backbonename %s now...' % args.backbonename)
    if use_cuda: model = model.cuda()
    # prepare optimizer
    optimizer_cfg = cfg.OPTIMIZER_CFG[cfg.OPTIMIZER_CFG['type']]
    start_epoch = 1
    end_epoch = optimizer_cfg['max_epochs']
    learning_rate_idx = 0
    if optimizer_cfg['is_use_warmup']:
        learning_rate = optimizer_cfg['learning_rates'][learning_rate_idx] / 3
    else:
        learning_rate = optimizer_cfg['learning_rates'][learning_rate_idx]
    if cfg.OPTIMIZER_CFG['type'] == 'sgd':
        sgd_builder_cfg = {
                            'learning_rate': learning_rate,
                            'momentum': optimizer_cfg['momentum'],
                            'weight_decay': optimizer_cfg['weight_decay']
                        }
        optimizer = SGDBuilder(model, sgd_builder_cfg, True)
    else:
        raise ValueError('Unsupport optimizer type %s now...' % cfg.OPTIMIZER_CFG['type'])
    # check checkpoints path
    if args.checkpointspath:
        checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
        model.load_state_dict(checkpoints['model'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        start_epoch = checkpoints['epoch'] + 1
        for epoch in range(1, start_epoch):
            if epoch in optimizer_cfg['adjust_lr_epochs']:
                learning_rate_idx += 1
    # data parallel
    if is_multi_gpus: model = nn.DataParallel(model)
    # print config
    logger_handle.info('Dataset used: %s, Number of images: %s' % (args.datasetname, len(dataset)))
    logger_handle.info('Backbone used: %s' % args.backbonename)
    logger_handle.info('Checkpoints used: %s' % args.checkpointspath)
    logger_handle.info('Config file used: %s' % cfg_file_path)
    # train
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    for epoch in range(start_epoch, end_epoch+1):
        # --set train mode
        if is_multi_gpus:
            model.module.setTrain()
        else:
            model.setTrain()
        # --adjust learning rate
        if epoch in optimizer_cfg['adjust_lr_epochs']:
            learning_rate_idx += 1
            adjustLearningRate(optimizer=optimizer, target_lr=optimizer_cfg['learning_rates'][learning_rate_idx], logger_handle=logger_handle)
        # --log info
        logger_handle.info('Start epoch %s, learning rate is %s...' % (epoch, optimizer_cfg['learning_rates'][learning_rate_idx]))
        # --train epoch
        for batch_idx, samples in enumerate(dataloader):
            if (epoch == 1) and (optimizer_cfg['is_use_warmup']) and (batch_idx <= optimizer_cfg['num_warmup_steps']):
                assert learning_rate_idx == 0, 'BUGS may exist...'
                target_lr = optimizer_cfg['learning_rates'][learning_rate_idx] / 3
                target_lr += (optimizer_cfg['learning_rates'][learning_rate_idx] - optimizer_cfg['learning_rates'][learning_rate_idx] / 3) * batch_idx / optimizer_cfg['num_warmup_steps']
                adjustLearningRate(optimizer=optimizer, target_lr=target_lr)
            optimizer.zero_grad()
            imageids, images, proposals, proposals_scores, gt_labels = samples
            output = model(x=images.type(FloatTensor), proposals=proposals.type(FloatTensor), proposals_scores=proposals_scores.type(FloatTensor), targets=gt_labels.type(FloatTensor))
            preds_cls, loss_cls = output
            loss = loss_cls.mean()
            logger_handle.info('[EPOCH]: %s/%s, [BATCH]: %s/%s, [LEARNING_RATE]: %s, [DATASET]: %s \n\t [LOSS]: loss_cls %.4f, total %.4f' % \
                               (epoch, end_epoch, (batch_idx+1), len(dataloader), optimizer_cfg['learning_rates'][learning_rate_idx], args.datasetname, loss_cls.mean().item(), loss.item()))
            loss.backward()
            clipGradients(model.parameters(), optimizer_cfg['grad_clip_max_norm'], optimizer_cfg['grad_clip_norm_type'])


'''run'''
if __name__ == '__main__':
    train()