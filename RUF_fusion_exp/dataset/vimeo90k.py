import glob
import random
import torch
import os.path as op
import numpy as np
import numpy
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv, ndarray2img
from torchvision import utils as vutils

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class Vimeo90KDataset(data.Dataset):
    """Vimeo-90K dataset.

    For training data: LMDB is adopted. See create_lmdb for details.

    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            '/data1/zz/data_wl/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            '/data1/zz/data_wl/',
            self.opts_dict['lq_path']
        )
        self.ref_root = op.join(
            '/data1/zz/data_wl/',
            self.opts_dict['ref_path']
        )
        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root,
            'meta_info.txt'
        )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root,
            self.gt_root,
            self.ref_root
        ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt', 'ref']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            # self.neighbor_list = [i + 1 for i in range(nfs)]
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        # get the neighboring GQ frames
        img_gt_path = f'{clip}/{seq}/im4.png'
        gt_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(gt_bytes)  # (H W 1)
        img_gt = numpy.squeeze(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lq = numpy.squeeze(img_lq)
            img_lqs.append(img_lq)

        # get the neighboring ref frames
        img_ref_path = f'{clip}/{seq}/im4.png'
        ref_bytes = self.file_client.get(img_ref_path, 'ref')
        img_ref = _bytes2img(ref_bytes)  # (H W 1)
        img_ref = numpy.squeeze(img_ref)

        # ==========
        # data augmentation
        # ==========
        # randomly crop
        img_gt, img_lqs, img_ref = paired_random_crop(
            img_gt, img_lqs, img_ref, gt_size, img_lq_path
        )

        # flip, rotate
        img_lqs.append(img_ref)
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        # img_two = img_lqs.append(img_ref)
        img_results = augment(
            img_lqs, hflip=True, rotation=True
        )
        # img_results_one = augment(
        #     img_two, self.opts_dict['use_flip'], self.opts_dict['use_rot']
        # )
        # to tensor
        img_results = totensor(img_results)
        # img_results_one = totensor(img_results_one)
        img_lqs = torch.stack(img_results[0:-2], dim=0)
        img_ref = img_results[-2]
        img_gt = img_results[-1]
        # img_ref = totensor(img_ref)
        # t, _, _, _  = img_lqs.shape
        # lqs_1 = img_lqs[0]
        # lqs_1 = lqs_1.clone().detach()
        # lqs_1 = lqs_1.to(torch.device('cpu'))
        # lqs_1_filename = "/home/zouzizhuang/weiliu/R3N-main/lqs_1.jpg"
        # vutils.save_image(lqs_1, lqs_1_filename)

        # lqs_2 = img_lqs[1]
        # lqs_2 = lqs_2.clone().detach()
        # lqs_2 = lqs_2.to(torch.device('cpu'))
        # lqs_2_filename = "/home/zouzizhuang/weiliu/R3N-main/lqs_2.jpg"
        # vutils.save_image(lqs_2, lqs_2_filename)

        # lqs_3 = img_lqs[2]
        # lqs_3 = lqs_3.clone().detach()
        # lqs_3 = lqs_3.to(torch.device('cpu'))
        # lqs_3_filename = "/home/weiliu/project-pycharm/R3N-main_2/pic/lqs_3.jpg"
        # vutils.save_image(lqs_3, lqs_3_filename)

        # lqs_4 = img_lqs[3]
        # lqs_4 = lqs_4.clone().detach()
        # lqs_4 = lqs_4.to(torch.device('cpu'))
        # lqs_4_filename = "/home/zouzizhuang/weiliu/R3N-main/lqs_4.jpg"
        # vutils.save_image(lqs_4, lqs_4_filename)

        # lqs_5 = img_lqs[4]
        # lqs_5 = lqs_5.clone().detach()
        # lqs_5 = lqs_5.to(torch.device('cpu'))
        # lqs_5_filename = "/home/zouzizhuang/weiliu/R3N-main/lqs_5.jpg"
        # vutils.save_image(lqs_5, lqs_5_filename)

        # gt = img_gt.clone().detach()
        # gt = gt.to(torch.device('cpu'))
        # gt_filename = "/home/weiliu/project-pycharm/R3N-main_2/pic/gt.jpg"
        # vutils.save_image(gt, gt_filename)

        # ref = img_ref.clone().detach()
        # ref = ref.to(torch.device('cpu'))
        # ref_filename = "/home/weiliu/project-pycharm/R3N-main_2/pic/ref.jpg"
        # vutils.save_image(ref, ref_filename)

        # img_ref = img_results_one[-1]
        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            'ref': img_ref # ([RGB] H W)
        }

    def __len__(self):
        return len(self.keys)

