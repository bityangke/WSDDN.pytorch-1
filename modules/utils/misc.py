'''
Function:
    some utils functions
Author:
    Charles
'''
import os
import torch
import logging
from torch.nn.utils import clip_grad


'''log function.'''
class Logger():
    def __init__(self, logfilepath, **kwargs):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(logfilepath),
                                      logging.StreamHandler()])
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    @staticmethod
    def debug(message):
        Logger.log(logging.DEBUG, message)
    @staticmethod
    def info(message):
        Logger.log(logging.INFO, message)
    @staticmethod
    def warning(message):
        Logger.log(logging.WARNING, message)
    @staticmethod
    def error(message):
        Logger.log(logging.ERROR, message)


'''filter small boxes'''
def filterSmallBoxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask


'''load class labels.'''
def loadclsnames(clsnamespath):
    names = []
    for line in open(clsnamespath):
        if line.strip('\n'):
            names.append(line.strip('\n'))
    return names


'''clip gradient'''
def clipGradients(params, max_norm=35, norm_type=2):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        clip_grad.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)


'''save checkpoints'''
def saveCheckpoints(state_dict, savepath, logger_handle):
    logger_handle.info('Saving state_dict in %s...' % savepath)
    torch.save(state_dict, savepath)
    return True


'''load checkpoints'''
def loadCheckpoints(checkpointspath, logger_handle):
    logger_handle.info('Loading checkpoints from %s...' % checkpointspath)
    checkpoints = torch.load(checkpointspath)
    return checkpoints


'''adjust learning rate'''
def adjustLearningRate(optimizer, target_lr, logger_handle=None):
    if logger_handle is not None:
        logger_handle.info('Adjust learning rate to %s...' % str(target_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
    return True


'''check the existence of dirpath'''
def checkDir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        return False
    return True