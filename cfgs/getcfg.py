'''
Function:
    used to get config file for specified dataset and backbone.
Author:
    Charles
'''
def getCfgByDatasetAndBackbone(datasetname, backbonename):
    if [datasetname, backbonename] == ['voc07', 'vgg16']:
        import cfgs.cfg_voc_vgg16 as cfg
        cfg_file_path = 'cfgs/cfg_voc07_vgg16'
    else:
        raise ValueError('Can not find cfg file for dataset <%s> and backbone <%s>...' % (datasetname, backbonename))
    return cfg, cfg_file_path