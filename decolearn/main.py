import os
import json
import datetime
import fire

import numpy as np
import torch
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main(gpu_index, is_optimize_regis):

    with open('config.json') as File:
        config = json.load(File)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    config['method']['proposed']['is_optimize_regis'] = str2bool(is_optimize_regis)
    config['setting']['save_folder'] = datetime.datetime.now().strftime("%m%d%H%M") + '_decolearn_is_optimize_regis=[%s]' % (str(is_optimize_regis))

    from torch_util.module import cvpr2018_net as voxelmorph
    from torch_util.module import EDSR, DeepUnfolding
    from method import DeCoLearn as DeCoLearn
    from dataset.modl import load_synthetic_MoDL_dataset

    nf_enc = config['module']['regis']['nf_enc']
    nf_dec = config['module']['regis']['nf_dec']

    '''
    recon_module = EDSR(
            n_resblocks=config['module']['recon']['EDSR']['n_resblocks'],
            n_feats=config['module']['recon']['EDSR']['n_feats'],
            res_scale=config['module']['recon']['EDSR']['res_scale'],
            in_channels=2,
            out_channels=2,
            dimension=2,
        )
    '''

    #muList = [0.75, 0.5, 0.35]
    muList = [0.65]

    for i in range(len(muList)):
        recon_module = DeepUnfolding(5, muList[i])
        regis_module = voxelmorph([256, 240], nf_enc, nf_dec)

        method_dict = {
            'DeCoLearn': DeCoLearn,
        }

        load_dataset_fn = lambda baseline_method: load_synthetic_MoDL_dataset(
                root_folder=config['dataset']['root_path'],
                mask_type=config['dataset']['mask_type'],
                mask_fold=config['dataset']['mask_fold'],
                input_snr=config['dataset']['input_snr'],
                nonlinear_P=config['dataset']['synthetic']['P'],
                nonlinear_sigma=config['dataset']['synthetic']['sigma'],
                nonlinear_theta=config['dataset']['synthetic']['theta'],
                translation=config['dataset']['synthetic']['translation'],
                rotate=config['dataset']['synthetic']['rotate'],
                scale=config['dataset']['synthetic']['scale'],
                mul_coil = config['dataset']['multi_coil'])


        method_dict[config['setting']['method']].train(
                    load_dataset_fn=load_dataset_fn,
                    recon_module=recon_module,
                    regis_module=regis_module,
                    config=config
        )

        method_dict[config['setting']['method']].test(
                    load_dataset_fn=load_dataset_fn,
                    recon_module=recon_module,
                    regis_module=regis_module,
                    config=config
        )


if __name__ == '__main__':
    fire.Fire(main)

