from operator import gt
import re
import torch
import numpy as np
from collections import OrderedDict
from RUF_test import R3N
import utils
from tqdm import tqdm
import os
import cv2
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math
import torch.nn as nn

# ./ffmpeg -s 1920x1080 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/ParkScene_1920x1080_240.yuv /home/newdata/data/test_ref_qp27_real/001/%3d.png
# ./ffmpeg -s 832x480 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BQMall_832x480_600.yuv /home/newdata/data/test_ref_qp27_real/002/%3d.png
# ./ffmpeg -s 1920x1080 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/Kimono_1920x1080_240.yuv /home/newdata/data/test_ref_qp27_real/003/%3d.png
# ./ffmpeg -s 1280x720 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/KristenAndSara_1280x720_600.yuv /home/newdata/data/test_ref_qp27_real/004/%3d.png
# ./ffmpeg -s 416x240 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/RaceHorses_416x240_300.yuv /home/newdata/data/test_ref_qp27_real/005/%3d.png
# ./ffmpeg -s 832x480 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BasketballDrillText_832x480_500.yuv /home/newdata/data/test_ref_qp27_real/006/%3d.png
# ./ffmpeg -s 1280x720 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/SlideEditing_1280x720_300.yuv /home/newdata/data/test_ref_qp27_real/007/%3d.png
# ./ffmpeg -s 832x480 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/PartyScene_832x480_500.yuv /home/newdata/data/test_ref_qp27_real/008/%3d.png
# ./ffmpeg -s 1280x720 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/SlideShow_1280x720_500.yuv /home/newdata/data/test_ref_qp27_real/009/%3d.png
# ./ffmpeg -s 1920x1080 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/Cactus_1920x1080_500.yuv /home/newdata/data/test_ref_qp27_real/010/%3d.png
# ./ffmpeg -s 416x240 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BlowingBubbles_416x240_500.yuv /home/newdata/data/test_ref_qp27_real/011/%3d.png
# ./ffmpeg -s 1280x720 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/FourPeople_1280x720_600.yuv /home/newdata/data/test_ref_qp27_real/012/%3d.png
# ./ffmpeg -s 1280x720 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/Johnny_1280x720_600.yuv /home/newdata/data/test_ref_qp27_real/013/%3d.png
# ./ffmpeg -s 2560x1600 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/PeopleOnStreet_2560x1600_150.yuv /home/newdata/data/test_ref_qp27_real/014/%3d.png
# ./ffmpeg -s 416x240 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BQSquare_416x240_600.yuv /home/newdata/data/test_ref_qp27_real/015/%3d.png
# ./ffmpeg -s 832x480 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/RaceHorses_832x480_300.yuv /home/newdata/data/test_ref_qp27_real/016/%3d.png
# ./ffmpeg -s 1920x1080 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BasketballDrive_1920x1080_500.yuv /home/newdata/data/test_ref_qp27_real/017/%3d.png
# ./ffmpeg -s 2560x1600 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/Traffic_2560x1600_150.yuv /home/newdata/data/test_ref_qp27_real/018/%3d.png
# ./ffmpeg -s 1024x768 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/ChinaSpeed_1024x768_500.yuv /home/newdata/data/test_ref_qp27_real/019/%3d.png
# ./ffmpeg -s 832x480 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BasketballDrill_832x480_500.yuv /home/newdata/data/test_ref_qp27_real/020/%3d.png
# ./ffmpeg -s 416x240 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BasketballPass_416x240_500.yuv /home/newdata/data/test_ref_qp27_real/021/%3d.png
# ./ffmpeg -s 1920x1080 -i /home/newdata/data/test_25_raw/test_22_raw_compress/qp27/BQTerrace_1920x1080_600.yuv /home/newdata/data/test_ref_qp27_real/022/%3d.png

# torch.cuda.set_device(2)

# /home/zouzizhuang/weiliu/R3N-main/exp/1x2_qp27_R3N_stdf_refqp32/ckp_300000.pth   /home/zouzizhuang/weiliu/R3N-main_1/exp/1x2_qp22_R3N_stdf_refqp27/ckp_300000.pth

# video_name = np.array(['ParkScene_1920x1080_240', 'BQMall_832x480_600', 'Kimono_1920x1080_240', 'KristenAndSara_1280x720_600', 'RaceHorses_416x240_300', 'BasketballDrillText_832x480_500', 'SlideEditing_1280x720_300',
#                         'PartyScene_832x480_500', 'SlideShow_1280x720_500', 'Cactus_1920x1080_500', 'BlowingBubbles_416x240_500', 'FourPeople_1280x720_600', 'Johnny_1280x720_600', 
#                         'PeopleOnStreet_2560x1600_150', 'BQSquare_416x240_600', 'RaceHorses_832x480_300', 'BasketballDrive_1920x1080_500', 'Traffic_2560x1600_150', 'ChinaSpeed_1024x768_500', 'BasketballDrill_832x480_500',
#                         'BasketballPass_416x240_500', 'BQTerrace_1920x1080_600'])
# # video_name = np.array([
# #                         'RaceHorses_832x480_300', 'Traffic_2560x1600_150', 'ChinaSpeed_1024x768_500', 'BasketballDrill_832x480_500',
# #                         'BasketballPass_416x240_500'])
# video_num = np.array(['001', '002', '003', '004', '005', '006', '007',
#                         '008', '009', '010', '011', '012', '013', 
#                         '014', '015', '016', '017', '018', '019', '020',
#                         '021', '022'])
# # video_num = np.array([
# #                         '016', '018', '019', '020',
# #                         '021'])
# ckp_path = '/8t/dataset_wl/exp/RUF_V1_qp32_refqp37/ckp_280000.pth'
# scene_name = '013'
# raw_yuv_path = '/data1/zz/data_wl/dataset_R3N/test_25_raw_pic/' + scene_name
# lq_yuv_path = '/data1/zz/data_wl/dataset_R3N/bicubic_test_qp32_data_pic/' + scene_name
# ref_yuv_path = '/data1/zz/data_wl/dataset_R3N/1x2_test_ref_pic/test_ref_qp37/' + scene_name
# # w, h = 1920, 1080
# nfs = 600


def main(video_num, video_name, H, W, Image_LQlist, Image_HQlist, Image_Reflist, ckp_path, nfs):
    model = R3N(H=H, W=W)
    msg = f'loading model {ckp_path}...'
    print(msg)
    # , map_location='cpu'
    # checkpoint = torch.load(ckp_path, map_location={'cuda:0': 'cuda:1'})
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    criterion = utils.PSNR()
    unit = 'dB'

    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()

    for idx in range(nfs):
        # if idx == 11:
        idx_list = list(range(idx - 2, idx + 3))
        idx_list = np.clip(idx_list, 0, nfs - 1)
        input_data = []
        print(idx_list)
        for idx_ in idx_list:
            input_data.append(Image_LQlist[idx_])
        # input_data = torch.from_numpy(np.array(input_data))
        # input_data = input_data.permute(0, 3, 1, 2).contiguous()
        # input_data = torch.unsqueeze(input_data, 0).cuda()

        input_data = totensor(input_data)
        input_data = torch.stack(input_data, dim=0)
        input_data = torch.unsqueeze(input_data, 0).cuda()
        input_data = input_data.squeeze(0).permute(0, 2, 3, 1)
        input_data = input_data.cpu().numpy()[:, :, :, ::-1]
        input_data = bgr2ycbcr(input_data)
        input_data = torch.tensor(input_data)
        input_data = torch.unsqueeze(input_data, 1).cuda()
        input_data = torch.unsqueeze(input_data, 0).cuda()
        # cc = input_data[0, 1, :, :, :]
        # SR_img = transforms.ToPILImage()(cc.cpu())
        # SR_img.save(f'/home/weiliu/project-pycharm/stdf-pytorch/lq_{idx}.png')
        # print(input_data.size())
        ref_data = totensor(Image_Reflist[idx])
        ref_data = ref_data.squeeze(0).permute(1, 2, 0)
        ref_data = ref_data.cpu().numpy()[:, :, ::-1]
        ref_data = bgr2ycbcr(ref_data)
        ref_data = torch.tensor(ref_data)
        ref_data = torch.unsqueeze(ref_data, 0).cuda()
        ref_data = torch.unsqueeze(ref_data, 0).cuda()

        ref_data_raw = bgr2ycbcr(Image_Reflist[idx])
        ref_data_raw = ref_data_raw * 255
        # ref_data = torch.stack(ref_data, dim=0)

        # ref_data = torch.from_numpy(Image_Reflist[idx]).cuda()
        # ref_data = ref_data.permute(2, 0, 1).contiguous()
        # ref_data = torch.unsqueeze(ref_data, 0).cuda()
        # bb = ref_data[0, 1, :, :]
        # SR_img = transforms.ToPILImage()(bb.cpu())
        # SR_img.save(f'/home/zouzizhuang/weiliu/R3N-main/fea_pic/ref_{idx}.png')
        # torch.Size([1, 240, 3, 536, 960])
        with torch.no_grad():
            print(input_data.shape)
            enhanced_frm, up_frm, _, _ = model(input_data, ref_data)
            # torch.Size([1, 3, 480, 832])
            enhanced_frm = torch.clip(enhanced_frm, 0, 1)
            # aa = enhanced_frm[0, :, :, :]
            # SR_img = transforms.ToPILImage()(aa.cpu())
            # SR_img.save(f'/home/zouzizhuang/weiliu/R3N-main/enh_pic/enh_{idx}.png')
        # enhanced_frm = enhanced_frm.squeeze(0).permute(1, 2, 0)
        # enhanced_frm = enhanced_frm.cpu().numpy()[:, :, ::-1]
        # enhanced_frm = bgr2ycbcr(enhanced_frm)
        enhanced_frm = enhanced_frm.cpu().numpy()
        enhanced_frm = enhanced_frm * 255

        # up_frm = up_frm.squeeze(0).permute(1, 2, 0)
        # up_frm = up_frm.cpu().numpy()[:, :, ::-1]
        # up_frm = bgr2ycbcr(up_frm)
        up_frm = up_frm.cpu().numpy()
        up_frm = up_frm * 255
        # aa = enhanced_frm[0, 1, :, :]
        # SR_img = transforms.ToPILImage()(aa.cpu())
        # SR_img.save(f'/home/zouzizhuang/weiliu/R3N-main/enh_pic/enh_{idx}.png')

        gt_frm = bgr2ycbcr(Image_HQlist[idx])
        gt_frm = gt_frm * 255

        # torch.Size([1, 3, 480, 832])
        # mm = totensor(Image_HQlist[idx])
        # mm = mm[:, :, :]
        # SR_img = transforms.ToPILImage()(mm.cpu())
        # SR_img.save(f'/home/zouzizhuang/weiliu/R3N-main/gt_pic/gt_{idx}.png')

        # batch_ori = criterion(input_data[0, 2, ...], gt_frm)
        if idx % 4 != 0:
            batch_perf = calculate_psnr(enhanced_frm, gt_frm)
            batch_perf_up = calculate_psnr(up_frm, gt_frm)
            ori_psnr_counter.accum(volume=batch_perf_up)

            # batch_ori = 0
            enh_psnr_counter.accum(volume=batch_perf)
            # display
            pbar.set_description(
                "{:s}-{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                    .format(video_num, video_name, batch_perf_up, unit, batch_perf, unit)
            )
            pbar.update()
        if idx % 4 == 0:
            batch_perf = calculate_psnr(ref_data_raw, gt_frm)
            batch_perf_up = calculate_psnr(up_frm, gt_frm)
            ori_psnr_counter.accum(volume=batch_perf_up)
            # batch_ori = 0
            enh_psnr_counter.accum(volume=batch_perf)
            # display
            pbar.set_description(
                "{:s}-{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                    .format(video_num, video_name, batch_perf_up, unit, batch_perf, unit)
            )
            pbar.update()
    pbar.close()
    ori_ = ori_psnr_counter.get_ave()

    # ori_ = 0
    enh_ = enh_psnr_counter.get_ave()
    print('{:s}-{:s}:ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        video_num, video_name, ori_, unit, enh_, unit, (enh_ - ori_), unit
    ))

    print('> done.')
    return enh_, ori_

def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def totensor(imgs, opt_bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        opt_bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, opt_bgr2rgb, float32):
        if img.shape[2] == 3 and opt_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, opt_bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, opt_bgr2rgb, float32)


def concat_image(lq_yuv_path, raw_yuv_path, ref_yuv_path, img_path_list, nfs):
    Image_LQlist = []
    Image_HQlist = []
    Image_Reflist = []
    for idx in range(nfs):
        lq_path = lq_yuv_path + '/' + img_path_list[idx]

        Image_LQlist.append(cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.)
        HQ_path = raw_yuv_path + '/' + img_path_list[idx]

        Image_HQlist.append(cv2.imread(HQ_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.)
        ref_path = ref_yuv_path + '/' + img_path_list[idx]

        Image_Reflist.append(cv2.imread(ref_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.)
    return Image_LQlist, Image_HQlist, Image_Reflist


def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]

    # img1是超分辨率后的图像但是好像只有y通道

    img1 = prediction.astype(np.float64)
    # cv2.imwrite('./out/33.PNG', img1)

    img2 = target.astype(np.float64)
    # cv2.imwrite('./out/333.PNG', img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

class val(nn.Module):
    def __init__(self, ckp_num_pth):
        super().__init__()
        enh = []
        up = []
        unit = 'dB'
        video_name = np.array(['ParkScene_1920x1080_240', 'BQMall_832x480_600', 'Kimono_1920x1080_240', 'KristenAndSara_1280x720_600', 'RaceHorses_416x240_300', 'BasketballDrillText_832x480_500', 'SlideEditing_1280x720_300',
                        'PartyScene_832x480_500', 'SlideShow_1280x720_500', 'Cactus_1920x1080_500', 'BlowingBubbles_416x240_500', 'FourPeople_1280x720_600', 'Johnny_1280x720_600',
                        'PeopleOnStreet_2560x1600_150', 'BQSquare_416x240_600', 'RaceHorses_832x480_300', 'BasketballDrive_1920x1080_500', 'Traffic_2560x1600_150', 'ChinaSpeed_1024x768_500', 'BasketballDrill_832x480_500',
                        'BasketballPass_416x240_500', 'BQTerrace_1920x1080_600'])

        video_num = np.array(['001', '002', '003', '004', '005', '006', '007',
                                '008', '009', '010', '011', '012', '013',
                                '014', '015', '016', '017', '018', '019', '020',
                                '021', '022'])

        video_name = np.array(['KristenAndSara_1280x720_600'])

        video_num = np.array(['004'])

        ckp_path = ckp_num_pth
        for i in range(0, len(video_num)):
            raw_yuv_path = '/home/newdata/data/test_25_raw_pic/' + video_num[i]
            lq_yuv_path = '/home/newdata/data/downsample_x2_compress/qp32/bicubic_test_qp32_data_pic/' + video_num[i]
            ref_yuv_path = '/home/newdata/data/ref/1x2_ref_pic/test_ref_qp37/' + video_num[i]

            list_IMAGE = os.listdir(lq_yuv_path)
            list_IMAGE.sort()
            nfs = len(list_IMAGE)

            Image_LQlist, Image_HQlist, Image_Reflist = concat_image(lq_yuv_path, raw_yuv_path, ref_yuv_path, list_IMAGE, nfs)
            H = Image_LQlist[i].shape[0]
            W = Image_LQlist[i].shape[1]
            enh_value, up_value = main(video_num[i], video_name[i], H, W, Image_LQlist, Image_HQlist, Image_Reflist, ckp_path, nfs)
            enh.append(enh_value)
            up.append(up_value)
        all = 0
        all_1 = 0
        for i in range(0, len(video_name)):
            print('{:s}-{:s}:  [{:.3f}] {:s} -> [{:.3f}] {:s} --enh:[{:.3f}]'.format(video_num[i], video_name[i], up[i], unit, enh[i], unit, (enh[i] - up[i])))
            all = all + enh[i]
            all_1 = all_1 + up[i]
        print('avg psnr: {:.3f}'.format(all/len(video_name)))
        return all/len(video_name), (all-all_1)/len(video_name)


if __name__ == '__main__':
    val(ckp_num_pth='/home/weiliu/project-pycharm/RUF_v2_y_1/exp/300_RUF_qp32_refqp37/ckp_300000.pth')
# aa = input_data[0,0,:,:,:]
# SR_img = transforms.ToPILImage()(aa.cpu())
# SR_img.save('xxx.png')

# 原始提取y，测量y
# 0 32.666241221647965 33.18595763513516
# 1 31.721222793876528 32.635293008224856
# 2 31.40791341471498 32.54738510736368
# 3 30.78578674252689 32.55001053384983
# 4 32.2069461726806 32.88743346400997
# 5 31.582763060150654 32.559544369514754
# 6 31.166607349241442 32.258857167337744
# 7 30.614612631263924 32.33439204610367
# 8 32.25002067193942 32.77688121560955
# 9 31.410965293835954 31.98293309034224

