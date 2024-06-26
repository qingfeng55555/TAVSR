import math
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import utils  # my tool box
import dataset
# from R3N_5 import R3N
from RUF import R3N
from collections import OrderedDict
# from Gaussian_downsample import gaussian_downsample
from test_all_video_module_cutsize import val, bgr2ycbcr, totensor

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_NTIRE_2022.yml',
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    # unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # ==========
    # init distributed training
    # ==========
    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank, 
            backend='nccl'
            )

    # TO-DO: load resume states if exists
    pass

    # ==========
    # create logger
    # ==========
    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        print("log_dir",log_dir)
        if not op.exists(log_dir):
            utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========

    pass

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up
    
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'], 
        radius=radius
        )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds, 
        num_replicas=opts_dict['train']['num_gpu'], 
        rank=rank, 
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
        )

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds, 
        opts_dict=opts_dict, 
        sampler=train_sampler, 
        phase='train',
        seed=opts_dict['train']['random_seed']
        )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
        opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
        opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    patch_size = opts_dict['dataset']['train']['gt_size']
    model = R3N(patch_size//2, patch_size//2)

    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ckp_path = 'exp/STDF_Multy_Pre_train/ckp_275000.pth'
    # checkpoint = torch.load(ckp_path)
    # state_dict = checkpoint['state_dict']
    # if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from1 {ckp_path}')
    # elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k  # add module
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    #     print(f'loaded from2 {ckp_path}')
    # else:  # the same way of training
    #     model.load_state_dict(state_dict)
    #     print(f'loaded from3 {ckp_path}')

    # define loss func
    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
        "Not implemented."
    loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])

    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(), 
        **opts_dict['train']['optim']
        )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
            'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer, 
            **opts_dict['train']['scheduler']
            )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    # criterion = utils.PSNR()


    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch

    see_loss = []
    model.train()
    num_iter_accum = start_iter

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # ckp_path = '/home/weiliu/project-pycharm/RUF_v2_y_1/exp/300_RUF_qp42_refqp47/ckp_260000.pth'
    # checkpoint = torch.load(ckp_path, map_location=device)
    # num_iter_accum= checkpoint["num_iter_accum"] + 1
    # model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # scaler = torch.cuda.amp.GradScaler()
    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        # train this epoch
        while train_data is not None:

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)  # (B T [RGB] H W)
            ref_data = train_data['ref'].to(rank) # (B [RGB] H W)
        
            # get data_y
            gt_data_y = gt_data.squeeze(0).permute(0, 2, 3, 1)
            gt_data_y = gt_data_y.cpu().numpy()[:, :, :, ::-1]
            gt_data_y = bgr2ycbcr(gt_data_y)
            gt_data_y = torch.tensor(gt_data_y)
            gt_data_y = torch.unsqueeze(gt_data_y, 1).cuda()

            lq_data_y = lq_data.squeeze(0).permute(0, 1, 3, 4, 2)
            lq_data_y = lq_data_y.cpu().numpy()[:, :, :, :, ::-1]
            lq_data_y = bgr2ycbcr(lq_data_y)
            lq_data_y = torch.tensor(lq_data_y)
            lq_data_y = torch.unsqueeze(lq_data_y, 2).cuda()

            ref_data_y = ref_data.squeeze(0).permute(0, 2, 3, 1)
            ref_data_y = ref_data_y.cpu().numpy()[:, :, :, ::-1]
            ref_data_y = bgr2ycbcr(ref_data_y)
            ref_data_y = torch.tensor(ref_data_y)
            ref_data_y = torch.unsqueeze(ref_data_y, 1).cuda()
            

            b, _, _, _, _  = lq_data.shape
            # input_data = torch.cat(
            #     [lq_data[:,:,i,...] for i in range(c)],
            #     dim=0
            #     )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            # gaussian_data = gaussian_downsample(input_data, 1).contiguous()
            # torch.Size([16, 1, 64, 64])
            # torch.Size([16, 5, 1, 32, 32])
            # torch.Size([16, 1, 64, 64])
            enhanced_data, enhanced_data_ref, enhanced_data_lqs = model(lq_data_y, ref_data_y)
            enhanced_data = torch.clip(enhanced_data,0,1)
            # get loss
            # print(enhanced_data.size())
            # print(gt_data.size())
            loss = torch.mean(torch.stack(
                [(0.5*(loss_func(enhanced_data[i], gt_data_y[i]))+0.25*(loss_func(enhanced_data_ref[i], gt_data_y[i]))+0.25*loss_func(enhanced_data_lqs[i], gt_data_y[i])) for i in range(b)]
                ))  # cal loss

            optimizer.zero_grad()  # zero grad
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                see_loss.append(loss_item)
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}]".format(
                        lr*1e4, loss_item
                        )
                    )
                print(msg)
                log_fp.write(msg + '\n')

            if ((num_iter_accum % (2*interval_val) == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0) and (num_iter_accum >= 50000):
            # if ((num_iter_accum % (interval_val) == 0) or \
            #     (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pth"
                    )
                state = {
                    'num_iter_accum': num_iter_accum, 
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # val
                if (num_iter_accum >= 50000) and (num_iter_accum % 10000 == 0):
                # if (num_iter_accum >= 100) and (num_iter_accum % 100 == 0):
                    avg_psnr, avg_enh = val(checkpoint_save_path)
                    msg = ("> avg_psnr: {:.4f}  avg_enh: {:.4f}").format(avg_psnr, avg_enh)
                    print(msg)
                    log_fp.write(msg + '\n')
                    log_fp.flush()

                # log
                msg = (
                    "> model saved at {:s}  avg_loss: {:.4f}\n"
                    ).format(
                        checkpoint_save_path, np.mean(see_loss)
                        )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

    # ==========
    # final log & close logger
    # ==========
    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
        print(msg)
        log_fp.write(msg + '\n')
        
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        
        log_fp.close()


if __name__ == '__main__':
    main()