"""Create LMDB only for training set of MFQEv2.

GT: non-overlapping 7-frame sequences extracted from 108 videos.
LQ: HM16.5-compised sequences.
key: assigned from 0000 to 9999.

NOTICE: MAX NFS OF LQ IS 300!!!

Sym-link MFQEv2 dataset root to ./data folder."""
import argparse
import os
import glob
import yaml
import os.path as op
from utils import make_y_lmdb_from_yuv

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='option_R3_mfqev2_1G.yml',
    help='Path to option YAML file.'
    )
args = parser.parse_args()

yml_path = args.opt_path
radius = 3  # must be 3!!! otherwise, you should change dataset.py


def create_lmdb_for_mfqev2():
    # video info
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = fp['dataset']['train']['root']
        gt_folder = fp['dataset']['train']['gt_folder']
        lq_folder = fp['dataset']['train']['lq_folder']
        ref_folder = fp['dataset']['train']['ref_folder']
        gt_path = fp['dataset']['train']['gt_path']
        lq_path = fp['dataset']['train']['lq_path']
        ref_path = fp['dataset']['train']['ref_path']
    gt_dir = op.join(root_dir, gt_folder)
    lq_dir = op.join(root_dir, lq_folder)
    ref_dir = op.join(root_dir, ref_folder)
    lmdb_gt_path = op.join(root_dir, gt_path)
    lmdb_lq_path = op.join(root_dir, lq_path)
    lmdb_ref_path = op.join(root_dir, ref_path)

    # scan all videos
    print('Scaning videos...')
    gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
    lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
    ref_video_list = sorted(glob.glob(op.join(ref_dir, '*.yuv')))

    msg = f'> {len(gt_video_list)} videos found.'
    print(msg)
    msg = f'> {len(lq_video_list)} videos found.'
    print(msg)
    msg = f'> {len(ref_video_list)} videos found.'
    print(msg)

    # generate LMDB for GT
    # print("Scaning GT frames (only center frames of each sequence)...")
    # frm_list = []
    # for gt_video_path in gt_video_list:
    #     nfs = int(gt_video_path.split('.')[-2].split('/')[-1].split('_')[-1])
    #     nfs = nfs if nfs <= 1500 else 1500 # !!!!!!
    #     num_seq = nfs // (2 * radius + 1)
    #     frm_list.append([radius + iter_seq * (2 * radius + 1) for iter_seq in range(num_seq)])
    # num_frm_total = sum([len(frms) for frms in frm_list])
    # msg = f'> {num_frm_total} frames found.'
    # print(msg)
    # key_list = []
    # video_path_list = []
    # index_frame_list = []
    # for iter_vid in range(len(gt_video_list)):
    #     frms = frm_list[iter_vid]
        
    #     for iter_frm in range(len(frms)):
    #         key_list.append('{:03d}/{:03d}/im4.png'.format(iter_vid + 1, iter_frm + 1))
    #         video_path_list.append(gt_video_list[iter_vid])
    #         index_frame_list.append(frms[iter_frm])
    
    # print("Writing LMDB for GT data...")
    # make_y_lmdb_from_yuv(
    #     video_path_list=video_path_list,
    #     index_frame_list=index_frame_list,
    #     key_list=key_list,
    #     lmdb_path=lmdb_gt_path,
    #     multiprocessing_read=True,
    #     only_y=False
    # )
    # print("> Finish.")

    # print("Scaning mask frames (only center frame of each sequence)...")
    # frm_list = []
    # frm_list_1 = []
    # for ref_video_path in ref_video_list:
    #     nfs = int(ref_video_path.split('.')[-2].split('/')[-1].split('_')[-1])
    #     nfs = nfs if nfs <= 1500 else 1500 # !!!!!!
    #     num_seq = nfs // (2 * radius + 1)
    #     # frm_list.append([radius + iter_seq * (2 * radius + 1) for iter_seq in range(num_seq)])
    #     for iter_seq in range(num_seq):
    #         if (radius + iter_seq * (2 * radius + 1))%4 <= 2:
    #             frm_list_1.append(((radius + iter_seq * (2 * radius + 1))//4*4))
    #         else:
    #             frm_list_1.append((((radius + iter_seq * (2 * radius + 1))//4+1)*4))
    #     frm_list.append(frm_list_1)
    #     frm_list_1 = []
    # num_frm_total = sum([len(frms) for frms in frm_list])
    # msg = f'> {num_frm_total} frames found.'
    # print(msg)
    # key_list = []
    # video_path_list = []
    # index_frame_list = []
    # for iter_vid in range(len(ref_video_list)):
    #     frms = frm_list[iter_vid]
        
    #     for iter_frm in range(len(frms)):
    #         key_list.append('{:03d}/{:03d}/im4.png'.format(iter_vid + 1, iter_frm + 1))
    #         video_path_list.append(ref_video_list[iter_vid])
    #         index_frame_list.append(frms[iter_frm])
    
    # print("Writing LMDB for mask data...")
    # make_y_lmdb_from_yuv(
    #     video_path_list=video_path_list,
    #     index_frame_list=index_frame_list,
    #     key_list=key_list,
    #     lmdb_path=lmdb_ref_path,
    #     multiprocessing_read=True,
    #     only_y=False
    # )
    # print("> Finish.")

    # generate LMDB for LQ
    print("Scaning LQ frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for lq_video_path in lq_video_list:
        nfs = int(lq_video_path.split('.')[-2].split('/')[-1].split('_')[-1])
        nfs = nfs if nfs <= 1500 else 1500  # !!!!!!
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input, (iter_seq + 1) \
                                    * len_input)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    key_list = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(lq_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            key_list.extend(['{:03d}/{:03d}/im{:d}.png'.format(iter_vid + 1, \
                                                               iter_seq + 1, i) for i in range(1, len_input + 1)])
            video_path_list.extend([lq_video_list[iter_vid]] * len_input)
            index_frame_list.extend(frm_seq[iter_seq])
    print(len(key_list))
    print("Writing LMDB for LQ data...")
    make_y_lmdb_from_yuv(
        video_path_list=video_path_list,
        index_frame_list=index_frame_list,
        key_list=key_list,
        lmdb_path=lmdb_lq_path,
        multiprocessing_read=True,
        only_y=False
    )
    print("> Finish.")

    # # sym-link
    # if not op.exists('data/MFQEv2'):
    #     if not op.exists('data/'):
    #         os.system("mkdir data/")
    #     os.system(f"ln -s {root_dir} ./data/MFQEv2")
    #     print("Sym-linking done.")
    # else:
    #     print("data/MFQEv2 already exists.")


if __name__ == '__main__':
    create_lmdb_for_mfqev2()