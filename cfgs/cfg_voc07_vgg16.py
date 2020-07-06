'''cfg file for voc07 dataset, vgg16 backbone'''


# backbone
BACKBONE_CFG = {
    'type': 'vgg16',
    'pretrained_model_path': '',
    'is_multi_gpus': True,
    'rois': {
                'type': ['roi_pool', 'roi_align'][-1],
                'size': (7, 7)
        }
}
# proposals
PROPOSALS_CFG = {
    'train': {
                'root_dir': 'proposals',
                'filename': 'EdgeBoxesVOC2007trainval.mat',
                'num_proposals': 2000
            },
    'test': {
                'root_dir': 'proposals',
                'filename': 'EdgeBoxesVOC2007test.mat',
                'num_proposals': 2000
            }
}
# dataset
DATASET_CFG = {
    'type': 'VOC07',
    'root_dir': '',
    'num_classes': 20,
    'num_workers': 8,
    'pin_memory': True,
    'batch_size': 8,
    'clsnamespath': 'names/voc.names',
    'use_color_jitter': False,
    'style': ['caffe', 'pytorch'][0],
    'scales': [480, 576, 688, 864, 1200],
    'max_num_gt_boxes': 50
}
# loss function
LOSS_CFG = {
            'cls_loss': {'type': 'binary_ce', 'binary_ce': {'size_average': True, 'weight': 1.0}}
        }
# optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
            'learning_rates': [1e-4, 1e-5],
            'max_epochs': 20,
            'adjust_lr_epochs': [11],
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'batch_size': DATASET_CFG['batch_size'],
            'is_use_warmup': True,
            'num_warmup_steps': 500,
            'grad_clip_max_norm': 35,
            'grad_clip_norm_type': 2
        },
}
# image size (max_len, min_len)
IMAGESIZE_DICT = {'LONG_SIDE': 1333, 'SHORT_SIDE': 800}
# record
RECORD_CFG = {
    'train': {
                'backupdir': 'wsddn_vgg16_trainbackup_voc07',
                'logfile': 'wsddn_vgg16_trainbackup_voc07/train.log',
                'save_interval': 1
            },
    'test': {
                'backupdir': 'wsddn_vgg16_testbackup_voc07',
                'logfile': 'wsddn_vgg16_testbackup_voc07/test.log',
                'bboxes_save_path': 'wsddn_vgg16_testbackup_voc07/wsddn_vgg16_detection_results_voc07.json'
            }
}