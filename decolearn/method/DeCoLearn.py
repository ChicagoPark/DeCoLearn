from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch

from torch_util.metrics import Mean, compare_psnr, compare_ssim, Stack, compare_snr

from torch_util.callback import CallbackList, BaseLogger, ModelCheckpoint, Tensorboard
from torch.utils.data import Dataset
from torch_util import losses
from torch_util.module import SpatialTransformer
import shutil
import datetime

from torch_util.common import dict2pformat, write_test, abs_helper, check_and_mkdir


class DictDataset(Dataset):
    def __init__(self, mode, data_dict, config):

        self.__data_dict = data_dict
        self.config = config

        assert mode in ['train', 'valid', 'test']

        self.__index_map = []

        n_slice = self.__data_dict['fixed_x'].shape[0]

        for slice_ in range(n_slice):
            self.__index_map.append([slice_])

        total_len = self.__index_map.__len__()
        print("total_len: ", total_len)

        if mode == 'train':
            self.__index_map = self.__index_map[0: 300]

        elif mode == 'valid':
            self.__index_map = self.__index_map[300: 360]

        else:
            self.__index_map = self.__index_map[360:]

    def __len__(self):
        return len(self.__index_map)

    def __getitem__(self, item):
        slice_, = self.__index_map[item]

        moved_x = self.__data_dict['moved_x'][slice_]
        moved_y_tran = self.__data_dict['moved_y_tran'][slice_]
        sensitivity_map = self.__data_dict['sensitivity_map'][slice_]

        moved_y = self.__data_dict['moved_y'][slice_]
        moved_mask = self.__data_dict['moved_mask'][slice_]

        fixed_x = self.__data_dict['fixed_x'][slice_]
        fixed_y_tran = self.__data_dict['fixed_y_tran'][slice_]

        fixed_y = self.__data_dict['fixed_y'][slice_]
        fixed_mask = self.__data_dict['fixed_mask'][slice_]

        return moved_x, moved_y_tran, sensitivity_map, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask


def train(
        load_dataset_fn,
        recon_module,
        regis_module,
        config
):
    train_dataset = DictDataset(
        mode='train',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[train_dataset] total_len: ", train_dataset.__len__())

    valid_dataset = DictDataset(
        mode='valid',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[valid_dataset] total_len: ", valid_dataset.__len__())

    ########################
    # Load Configuration
    ########################
    regis_batch = config['method']['proposed']['regis_batch']
    recon_batch = config['method']['proposed']['recon_batch']

    is_optimize_regis = config['method']['proposed']['is_optimize_regis']
    mul_coil = config['dataset']['multi_coil']
    training_type = config["train"]["training_type"]

    lambda_ = config['method']['lambda_']
    loss_regis_mse_COEFF = config['method']['loss_regis_mse']
    loss_regis_dice_COEFF = config['method']['loss_regis_dice']

    loss_recon_consensus_COEFF = config['method']['loss_recon_consensus']

    recon_lr, regis_lr = config['train']['recon_lr'], config['train']['regis_lr']
    recon_loss, regis_loss = config['train']['recon_loss'], config['train']['regis_loss']

    batch_size = config['train']['batch_size']

    num_workers = config['train']['num_workers']
    train_epoch = config['train']['train_epoch']
    verbose_batch = config['train']['verbose_batch']
    tensorboard_batch = config['train']['tensorboard_batch']
    checkpoint_epoch = config['train']['checkpoint_epoch']

    check_and_mkdir(config['setting']['root_path'])
    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'
    ########################
    # Dataset
    ########################

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    train_iter_total = int(train_dataset.__len__() / batch_size)

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)
    valid_iter_total = int(valid_dataset.__len__() / 1)

    sample_indices = [10, 20, 30]
    '''
    valid_sample_moved_x, valid_sample_moved_y_tran, valid_sensitivity_map, valid_sample_moved_y, valid_sample_moved_mask, \
        valid_sample_fixed_x, valid_sample_fixed_y_tran, valid_sample_fixed_y, valid_sample_fixed_mask \
        = (i.cuda() for i in next(iter(
            DataLoader(Subset(valid_dataset, sample_indices), batch_size=len(sample_indices)))))
    '''
    valid_sample_moved_x, valid_sample_moved_y_tran, valid_sensitivity_map, valid_sample_moved_y, valid_sample_moved_mask, \
    valid_sample_fixed_x, valid_sample_fixed_y_tran, valid_sample_fixed_y, valid_sample_fixed_mask \
        = (i.cuda() for i in next(iter(
        DataLoader(Subset(valid_dataset, sample_indices), batch_size=1))))

    print(valid_sample_moved_x.shape, valid_sample_moved_y_tran.shape, valid_sample_fixed_x.shape,
          valid_sample_fixed_y_tran.shape)

    image_init = {
        'groundtruth': abs_helper(valid_sample_fixed_x),
        'zero-filled': abs_helper(valid_sample_fixed_y_tran),
    }

    ########################
    # Metrics
    ########################
    metrics = Mean()

    ########################
    # Extra-Definition
    ########################
    regis_module.cuda()
    recon_module.cuda()

    sim_loss_fn = losses.ncc_loss
    grad_loss_fn = losses.gradient_loss
    mse_loss_fn = losses.mse_loss

    loss_fn_dict = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'smooth_l1': nn.SmoothL1Loss
    }

    recon_loss_fn = loss_fn_dict[recon_loss]()

    trf_232 = SpatialTransformer([256, 232])
    trf_240 = SpatialTransformer([256, 240])
    trf_232.cuda()
    trf_240.cuda()

    ########################
    # Begin Training
    ########################
    regis_optimizer = Adam(regis_module.parameters(), lr=regis_lr)
    recon_optimizer = Adam(recon_module.parameters(), lr=recon_lr)

    check_and_mkdir(file_path)
    regis_callbacks = CallbackList(callbacks=[
        BaseLogger(file_path=file_path),
        Tensorboard(file_path=file_path, per_batch=tensorboard_batch),
        ModelCheckpoint(file_path=file_path + 'regis_model/',
                        period=checkpoint_epoch,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max'])
    ])

    regis_callbacks.set_module(regis_module)
    regis_callbacks.set_params({
        'config': config,
        "lr": regis_lr,
        'train_epoch': train_epoch
    })

    recon_callbacks = CallbackList(callbacks=[
        ModelCheckpoint(file_path=file_path + 'recon_model/',
                        period=checkpoint_epoch,
                        monitors=['valid_psnr', 'valid_ssim'],
                        modes=['max', 'max']),
        BaseLogger(file_path=None)
    ])

    recon_callbacks.set_module(recon_module)
    recon_callbacks.set_params({
        'config': config,
        "lr": recon_lr,
        'train_epoch': train_epoch
    })

    regis_callbacks.call_train_begin_hook(image_init)
    recon_callbacks.call_train_begin_hook(image_init)

    global_batch = 1
    for global_epoch in range(1, train_epoch):

        regis_module.train()
        recon_module.train()

        iter_ = tqdm(train_dataloader, desc='Train [%.3d/%.3d]' % (global_epoch, train_epoch), total=train_iter_total)
        for i, train_data in enumerate(iter_):
            moved_x, moved_y_tran, sensitivity_map, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = \
                (i.cuda() for i in train_data)
            log_batch = {}

            regis_module.train()
            recon_module.train()

            if is_optimize_regis:

                for j in range(regis_batch):
                    fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                    moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                    fixed_y_tran_recon = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
                    moved_y_tran_recon = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

                    wrap_m2f, flow_m2f = regis_module(moved_y_tran_recon, fixed_y_tran_recon)

                    regis_recon_loss_m2f = sim_loss_fn(wrap_m2f, fixed_y_tran_recon)
                    regis_grad_loss_m2f = grad_loss_fn(flow_m2f)
                    regis_mse_loss_m2f = mse_loss_fn(wrap_m2f, fixed_y_tran_recon)

                    wrap_f2m, flow_f2m = regis_module(fixed_y_tran_recon, moved_y_tran_recon)

                    regis_recon_loss_f2m = sim_loss_fn(wrap_f2m, moved_y_tran_recon)
                    regis_grad_loss_f2m = grad_loss_fn(flow_f2m)
                    regis_mse_loss_f2m = mse_loss_fn(wrap_f2m, moved_y_tran_recon)

                    regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m
                    if lambda_ > 0:
                        regis_loss += lambda_ * (regis_grad_loss_m2f + regis_grad_loss_f2m)

                    if loss_regis_mse_COEFF > 0:
                        regis_loss += loss_regis_mse_COEFF * (regis_mse_loss_m2f + regis_mse_loss_f2m)


                    regis_optimizer.zero_grad()
                    regis_loss.backward()

                    #torch.nn.utils.clip_grad_value_(regis_module.parameters(), clip_value=1)
                    torch.nn.utils.clip_grad_value_(regis_module.parameters(), clip_value=0.5)
                    #torch.nn.utils.clip_grad_norm_(regis_module.parameters(), max_norm = 0.3)


                    regis_optimizer.step()

                    recon_module.zero_grad()
                    regis_module.zero_grad()

                    if j == (regis_batch - 1):
                        log_batch.update({
                            'registration_loss': regis_loss.item(),
                        })

            regis_module.train()
            recon_module.train()

            for j in range(recon_batch):
                fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
                moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                if is_optimize_regis:
                    fixed_y_tran_recon_abs = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
                    moved_y_tran_recon_abs = torch.nn.functional.pad(
                        torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

                    _, flow_m2f = regis_module(moved_y_tran_recon_abs, fixed_y_tran_recon_abs)
                    flow_m2f = flow_m2f[..., 4:-4]
                    wrap_m2f = torch.cat([trf_232(tmp, flow_m2f) for tmp in [
                        torch.unsqueeze(moved_y_tran_recon[:, 0], 1), torch.unsqueeze(moved_y_tran_recon[:, 1], 1)
                    ]], 1)
                    from torch_util.module import ftran, fmult
                    #wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(wrap_m2f.permute([0, 2, 3, 1]).contiguous())))
                    #wrap_y_m2f = fmult(wrap_m2f.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil)
                    wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (torch.view_as_complex(wrap_m2f.permute([0, 2, 3, 1]).contiguous()))))

                    _, flow_f2m = regis_module(fixed_y_tran_recon_abs, moved_y_tran_recon_abs)
                    flow_f2m = flow_f2m[..., 4:-4]
                    wrap_f2m = torch.cat([trf_232(tmp, flow_f2m) for tmp in [
                        torch.unsqueeze(fixed_y_tran_recon[:, 0], 1), torch.unsqueeze(fixed_y_tran_recon[:, 1], 1)
                    ]], 1)

                    #wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(wrap_f2m.permute([0, 2, 3, 1]).contiguous())))
                    #wrap_y_f2m = fmult(wrap_f2m.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil)
                    wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (torch.view_as_complex(wrap_f2m.permute([0, 2, 3, 1]).contiguous()))))

                else:
                    from torch_util.module import ftran, fmult
                    #wrap_y_m2f = fmult(wrap_m2f.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil)
                    #wrap_y_f2m = fmult(wrap_f2m.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil)
                    #wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))
                    #wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))
                    wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))))
                    wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))))

                recon_loss_m2f = recon_loss_fn(wrap_y_m2f, fixed_y)
                recon_loss_f2m = recon_loss_fn(wrap_y_f2m, moved_y)

                recon_loss = recon_loss_f2m + recon_loss_m2f
                '''
                recon_loss_consensus_fixed = recon_loss_fn(
                    fixed_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))), fixed_y)
                recon_loss_consensus_moved = recon_loss_fn(
                    moved_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))), moved_y)
                '''
                '''
                recon_loss_consensus_fixed = recon_loss_fn(
                    fmult(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, fixed_mask, mul_coil),
                    fixed_y)
                recon_loss_consensus_moved = recon_loss_fn(
                    fmult(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous(), sensitivity_map, moved_mask, mul_coil),
                    moved_y)
                '''
                recon_loss_consensus_fixed = recon_loss_fn(
                    fixed_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (
                        torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))), fixed_y)
                recon_loss_consensus_moved = recon_loss_fn(
                    moved_mask * torch.view_as_real(torch.fft.fft2(sensitivity_map * (
                        torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))), moved_y)

                if loss_recon_consensus_COEFF > 0:
                    recon_loss += loss_recon_consensus_COEFF * (recon_loss_consensus_fixed + recon_loss_consensus_moved)

                recon_optimizer.zero_grad()
                recon_loss.backward()

                #torch.nn.utils.clip_grad_value_(recon_module.parameters(), clip_value=1)
                torch.nn.utils.clip_grad_value_(recon_module.parameters(), clip_value=0.5)
                #torch.nn.utils.clip_grad_norm_(recon_module.parameters(), max_norm=0.5)

                recon_optimizer.step()

                recon_module.zero_grad()
                regis_module.zero_grad()

                if j == (recon_batch - 1):
                    # to remove the gray background
                    fixed_y_tran_recon[fixed_x == 0] = 0

                    log_batch.update({
                        'reconstruction_loss': recon_loss.item(),

                        'train_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'train_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),

                    })

            metrics.update_state(log_batch)

            if (verbose_batch > 0) and (global_batch % verbose_batch == 0):
                iter_.write(("Batch [%.7d]:" % global_batch) + dict2pformat(log_batch))
                iter_.update()

            regis_callbacks.call_batch_end_hook(log_batch, global_batch)
            recon_callbacks.call_batch_end_hook(log_batch, global_batch)
            global_batch += 1

        regis_module.eval()
        recon_module.eval()

        with torch.no_grad():

            iter_ = tqdm(valid_dataloader, desc='Valid [%.3d/%.3d]' % (global_epoch, train_epoch),
                         total=valid_iter_total)
            for i, valid_data in enumerate(iter_):
                moved_x, moved_y_tran, sensitivity_map, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = \
                    (i.cuda() for i in valid_data)

                fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)

                # added for registration visualization
                moved_y_tran_recon = recon_module(moved_y_tran, moved_mask, sensitivity_map, moved_y)

                fixed_x = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(fixed_x ** 2, dim=1, keepdim=True)), [4, 4])

                moved_x = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(moved_x ** 2, dim=1, keepdim=True)), [4, 4])

                fixed_y_tran_recon = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
                moved_y_tran_recon = torch.nn.functional.pad(
                    torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

                wrap_m2f, flow_m2f = regis_module(moved_y_tran_recon, fixed_y_tran_recon)
                wrap_f2m, flow_f2m = regis_module(fixed_y_tran_recon, moved_y_tran_recon)

                # to remove the gray background
                #fixed_y_tran_recon[fixed_x == 0] = 0

                if training_type == "pnp":
                    gammaList = recon_module.getGamma()
                    alphaList = recon_module.getAlpha()
                    log_batch = {
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'gamma0': gammaList[0].item(),
                        'gamma1': gammaList[1].item(),
                        'gamma2': gammaList[2].item(),
                        'gamma3': gammaList[3].item(),
                        'gamma4': gammaList[4].item(),
                        'alpha0': alphaList[0].item(),
                        'alpha1': alphaList[1].item(),
                        'alpha2': alphaList[2].item(),
                        'alpha3': alphaList[3].item(),
                        'alpha4': alphaList[4].item()
                    }
                elif training_type == "red":
                    # gammaList = recon_module.getGamma()
                    muList = recon_module.getMu()
                    log_batch = {
                        'mu': muList[0],
                        #'mu1': muList[1].item(),
                        #'mu2': muList[2].item(),
                        #'mu3': muList[3].item(),
                        #'mu4': muList[4].item(),
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),

                        #'valid_dice_m2f': compute_dice(moved_seg_warped_m2f, fixed_seg).item(),
                        #'valid_dice_f2m': compute_dice(fixed_seg_warped_f2m, moved_seg).item(),

                        'valid_fixed_snr': compare_snr(fixed_y_tran_recon, fixed_x).item(),
                        'valid_fixed_ssim': compare_ssim(fixed_y_tran_recon, fixed_x).item(),
                        'valid_fixed_psnr': compare_psnr(fixed_y_tran_recon, fixed_x).item(),

                        'valid_moved_snr': compare_snr(moved_y_tran_recon, moved_x).item(),
                        'valid_moved_ssim': compare_ssim(moved_y_tran_recon, moved_x).item(),
                        'valid_moved_psnr': compare_psnr(moved_y_tran_recon, moved_x).item(),

                        'valid_wrap_snr_m2f': compare_snr(wrap_m2f, fixed_x).item(),
                        'valid_wrap_ssim_m2f': compare_ssim(wrap_m2f, fixed_x).item(),
                        'valid_wrap_psnr_m2f': compare_psnr(wrap_m2f, fixed_x).item(),

                        'valid_wrap_snr_f2m': compare_snr(wrap_f2m, moved_x).item(),
                        'valid_wrap_ssim_f2m': compare_ssim(wrap_f2m, moved_x).item(),
                        'valid_wrap_psnr_f2m': compare_psnr(wrap_f2m, moved_x).item(),

                    }
                else:
                    log_batch = {
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                    }

                metrics.update_state(log_batch)

            valid_sample_fixed_y_tran_recon = recon_module(valid_sample_fixed_y_tran, valid_sample_fixed_mask,
                                                           sensitivity_map, valid_sample_fixed_y)

            # added
            valid_sample_moved_y_tran_recon = recon_module(valid_sample_moved_y_tran, valid_sample_moved_mask,
                                                           sensitivity_map, valid_sample_moved_y)



            valid_sample_moved_y_tran_recon = torch.nn.functional.pad(
                torch.sqrt(torch.sum(valid_sample_moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
            valid_sample_fixed_y_tran_recon = torch.nn.functional.pad(
                torch.sqrt(torch.sum(valid_sample_fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])


            valid_sample_wrap, valid_sample_flow = regis_module(valid_sample_moved_y_tran_recon,
                                                                valid_sample_fixed_y_tran_recon)



            valid_grids = create_standard_grid(valid_sample_flow)
            valid_grids = valid_grids.cuda()


            valid_grids = trf_240(valid_grids, valid_sample_flow)

            valid_wrap_norm = create_grid_norm(valid_sample_flow)

            '''

            train_sample_fixed_y_tran_recon = recon_module(train_sample_fixed_y_tran)
            train_sample_moved_y_tran_recon = recon_module(train_sample_moved_y_tran)

            train_sample_wrap, train_sample_flow = regis_module(train_sample_moved_y_tran_recon,
                                                                train_sample_fixed_y_tran_recon)

            train_grids = create_standard_grid(train_sample_flow)
            train_grids = train_grids.cuda()
            train_grids = trf(train_grids, train_sample_flow)

            train_wrap_norm = create_grid_norm(train_sample_flow)
            '''

        log_epoch = metrics.result()
        metrics.reset_state()

        image_epoch = {
            'prediction': abs_helper(valid_sample_fixed_y_tran_recon),

            'valid_sample_moved_y_tran_recon': valid_sample_moved_y_tran_recon,
            'valid_sample_fixed_y_tran_recon': valid_sample_fixed_y_tran_recon,
            "valid_sample_wrap": valid_sample_wrap,

            "valid_grids": valid_grids,
            "valid_wrap_norm": valid_wrap_norm,
        }

        regis_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)
        recon_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)
        '''
        with torch.no_grad():

            iter_ = tqdm(valid_dataloader, desc='Valid [%.3d/%.3d]' % (global_epoch, train_epoch),
                         total=valid_iter_total)
            for i, valid_data in enumerate(iter_):
                moved_x, moved_y_tran, sensitivity_map, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = \
                    (i.cuda() for i in valid_data)

                fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)

                # to remove the gray background
                fixed_y_tran_recon[fixed_x == 0] = 0

                if training_type == "pnp":
                    gammaList = recon_module.getGamma()
                    alphaList = recon_module.getAlpha()
                    log_batch = {
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'gamma0': gammaList[0].item(),
                        'gamma1': gammaList[1].item(),
                        'gamma2': gammaList[2].item(),
                        'gamma3': gammaList[3].item(),
                        'gamma4': gammaList[4].item(),
                        'alpha0': alphaList[0].item(),
                        'alpha1': alphaList[1].item(),
                        'alpha2': alphaList[2].item(),
                        'alpha3': alphaList[3].item(),
                        'alpha4': alphaList[4].item()
                    }
                elif training_type == "red":
                    #gammaList = recon_module.getGamma()
                    muList = recon_module.getMu()
                    log_batch = {
                        'mu0': muList[0].item(),
                        'mu1': muList[1].item(),
                        'mu2': muList[2].item(),
                        'mu3': muList[3].item(),
                        'mu4': muList[4].item(),
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item()
                    }
                else:
                    log_batch = {
                        'valid_ssim': compare_ssim(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                        'valid_psnr': compare_psnr(abs_helper(fixed_y_tran_recon), abs_helper(fixed_x)).item(),
                    }

                metrics.update_state(log_batch)

            valid_sample_fixed_y_tran_recon = recon_module(valid_sample_fixed_y_tran, valid_sample_fixed_mask, sensitivity_map, valid_sample_fixed_y)

        log_epoch = metrics.result()
        metrics.reset_state()

        image_epoch = {

            'prediction': abs_helper(valid_sample_fixed_y_tran_recon),

        }

        regis_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)
        recon_callbacks.call_epoch_end_hook(log_epoch, image_epoch, global_epoch)
        '''

def test(
        load_dataset_fn,
        recon_module: nn.Module,
        regis_module: nn.Module,
        config,
):
    test_dataset = DictDataset(
        mode='test',
        data_dict=load_dataset_fn(baseline_method=None),
        config=config)
    print("[test_dataset] total_len: ", test_dataset.__len__())

    file_path = config['setting']['root_path'] + config['setting']['save_folder'] + '/'

    recon_checkpoint = file_path + config['test']['recon_checkpoint']
    recon_module.load_state_dict(torch.load(recon_checkpoint))

    recon_module.cuda()
    recon_module.eval()

    metrics = Stack()
    images = Stack()

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        iter_ = tqdm(test_dataloader, desc='Test', total=len(test_dataset))
        for i, test_data in enumerate(iter_):
            moved_x, moved_y_tran, sensitivity_map, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = \
                (i.cuda() for i in test_data)
            fixed_y_tran_recon = recon_module(fixed_y_tran, fixed_mask, sensitivity_map, fixed_y)
            fixed_y_tran_recon = abs_helper(fixed_y_tran_recon)

            fixed_x = abs_helper(fixed_x)

            fixed_y_tran = abs_helper(fixed_y_tran)
            # to remove the gray background
            fixed_y_tran[fixed_x == 0] = 0
            fixed_y_tran_recon[fixed_x == 0] = 0

            log_batch = {
                'zero_filled_ssim': compare_ssim(fixed_y_tran, fixed_x).item(),
                'zero_filled_psnr': compare_psnr(fixed_y_tran, fixed_x).item(),

                'prediction_ssim': compare_ssim(fixed_y_tran_recon, fixed_x).item(),
                'prediction_psnr': compare_psnr(fixed_y_tran_recon, fixed_x).item(),

            }

            images_batch = {
                'groundtruth': fixed_x.detach().cpu().numpy(),
                'zero_filled': fixed_y_tran.detach().cpu().numpy(),
                'prediction': fixed_y_tran_recon.detach().cpu().numpy(),

            }

            metrics.update_state(log_batch)
            images.update_state(images_batch)

    save_path = file_path + 'test_' + datetime.datetime.now().strftime("%m%d%H%M") + \
        '_' + config['setting']['save_folder'] + \
        '_' + config['test']['recon_checkpoint'].replace('/', '_') + '/'

    check_and_mkdir(save_path)

    check_and_mkdir(save_path + 'recon_model/')
    shutil.copy(recon_checkpoint, save_path + 'recon_model/')

    print("Writing results....")
    write_test(log_dict=metrics.result(), img_dict=images.result(), save_path=save_path,
               is_save_mat=config['test']['is_save_mat'])


# Function for registration module visualization
# -------------------------------------

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import tifffile as tiff
import torch
from PIL import Image, ImageDraw
from skimage.measure import find_contours
import scipy.io as sio
import ants

def create_standard_grid(grid):
    n_batch, _, n_width, n_height = grid.shape

    img_ants = ants.from_numpy(np.zeros([n_width, n_height], dtype=np.float32))

    img_array = ants.create_warped_grid(image=img_ants, foreground=0, background=1).numpy(True)
    img_array = np.transpose(img_array, [2, 0, 1])

    img_array -= np.amin(img_array)
    img_array /= np.amax(img_array)

    if n_batch > 1:
        img_grid = torch.stack(n_batch * [torch.from_numpy(img_array)], 0)
    else:
        img_grid = torch.from_numpy(img_array).unsqueeze(0)

    return img_grid

def create_grid_norm(grid):
    n_batch, n_dim, n_width, n_height = grid.shape

    if n_dim == 2:
        norm = torch.sqrt(grid[:, 0, :, :] ** 2 + grid[:, 1, :, :] ** 2)
        norm = norm.unsqueeze(1)

    else:
        raise NotImplementedError("Only n_dim = 2 Supported")

    return norm
