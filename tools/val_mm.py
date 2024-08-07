import cv2
import datetime
import torchvision.utils as vutil
import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.pyt_utils import ensure_dir
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.models.modules.cfm import ExternalAttentionRectifyModule as EARM
from semseg.models.modules.cfm import CrossFusionModule as CFM

INSTANCE_FOLDER = '/home/gsn/icode/MSegmentation/DELIVER/result'

logger = get_logger()


# def get_my_labels(*args):
#     " r,g,b"
#     return np.array([
#         [44, 160, 44],  # asphalt
#         [31, 119, 180],  # concrete
#         [255, 127, 14],  # metal
#         [214, 39, 40],  # road marking
#         [140, 86, 75],  # fabric, leather
#         [127, 127, 127],  # glass
#         [188, 189, 34],  # plaster
#         [255, 152, 150],  # plastic
#         [23, 190, 207],  # rubber
#         [174, 199, 232],  # sand
#         [196, 156, 148],  # gravel
#         [197, 176, 213],  # ceramic
#         [247, 182, 210],  # cobblestone
#         [199, 199, 199],  # brick
#         [219, 219, 141],  # grass
#         [158, 218, 229],  # wood
#         [57, 59, 121],  # leaf
#         [107, 110, 207],  # water
#         [156, 158, 222],  # human body
#         [99, 121, 57]])  # sky
def get_my_labels(*args):
    " r,g,b"
    return np.array([
        [0, 0, 0],  # asphalt
        [31, 119, 180],  # concrete
        [255, 127, 14],  # metal
        [214, 39, 40],  # road marking
        [140, 86, 75],  # fabric, leather
        [127, 127, 127],  # glass
        [188, 189, 34],  # plaster
        [255, 152, 150],  # plastic
        [255,255,255]])  # sky
def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)

            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)

            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d_%d.png' % (base_name, index, 0))
    return image_name
def hook_func_d2o(module, input, output):
    """
      Hook function of register_forward_hook

      Parameters:
      -----------
      module: module of neural network
      input: input of module
      output: output of module
      """
    image_name = get_image_name_for_hook(module)
    image_name = image_name.rstrip('0.png')
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    # vutil.save_image(data, image_name, pad_value=0.5)
    for j in range(4):
        data1 = data[j]
        timestamp = datetime.datetime.now().strftime("%M-%S")
        savepath = image_name + timestamp + '.png'
        vutil.save_image(data1, savepath, pad_value=0.5)




def hook_func(module, input, output):
    """
    Hook function of register_forward_hook

    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    image_name = image_name.rstrip('0.png')
    for i in range(2):
        out = output[i]
        data = out.clone().detach()
        data = data.permute(1, 0, 2, 3)
        for j in range(2):
            data1 = data[j]
            timestamp = datetime.datetime.now().strftime("%M-%S")
            savepath = image_name + timestamp + ('%d.png' % i)
            vutil.save_image(data1, savepath, pad_value=0.5)


@torch.no_grad()
def evaluate(model, dataloader, device, bgm, writer):
    print('Evaluating...')
    # 这里临时添加了网络特征的可视化函数
    for name, module in model.named_modules():
        if isinstance(module, EARM):
            module.register_forward_hook(hook_func)
        if isinstance(module, CFM):
            module.register_forward_hook(hook_func_d2o)
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    item = 0

    for images, labels in tqdm(dataloader):
        item = item + 1
        images = [x.to(device) for x in images]
        # first one  =  last one
        if bgm == 1:
            labels = labels[0]
            labels = labels.to(device)
        else:
            labels = labels.to(device)

        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            if bgm == 1:
                preds, _, _ = model(images)
                preds = preds.softmax(dim=1)
            else:
                preds = model(images).softmax(dim=1)
        metrics.update(preds, labels)


        # 记录特征到TensorBoard
        # 获取某一层的特征
        # add by gsn for print result
        # ......
        # save_path = 'result'
        # if save_path is not None:
        #     pred = preds[0]
        #     pred = torch.argmax(pred, dim=0).cpu().numpy()
        #     ensure_dir(save_path)
        #     ensure_dir(save_path+'/result_color')
        #
        #     fn = str(item) + '.png'
        #
        #     # save colored result
        #     result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        #     class_colors = get_my_labels()
        #     palette_list = list(np.array(class_colors).flat)
        #     if len(palette_list) < 60:
        #         palette_list += [0] * (60 - len(palette_list))
        #     result_img.putpalette(palette_list)
        #     result_img.save(os.path.join(save_path+'/result_color', fn))
        #
        #     # save raw result
        #     cv2.imwrite(os.path.join(save_path, fn), pred*5)
        #     logger.info('Save the image ' + fn)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip, bgm):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        if bgm == 0:
            labels = labels.to(device)
        else:
            labels = [lab.to(device) for lab in labels]
            labels = labels[0]
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            if bgm == 0:
                logits = model(scaled_images)
            else:
                logits, _, _ = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                if bgm == 0:
                    logits = model(scaled_images)
                else:
                    logits, _, _ = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))
    bgm = cfg['MODEL']['BOUNDARY']

    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    writer = SummaryWriter(str(save_dir))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'], bgm)
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device, bgm, writer)

            table = {
                'Class': list(dataset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(eval_cfg['MODEL_PATH']))

        with open(eval_path, 'a+') as f:
            f.writelines(eval_cfg['MODEL_PATH'])
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)
    writer.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mcubes_rgbadn_next.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)