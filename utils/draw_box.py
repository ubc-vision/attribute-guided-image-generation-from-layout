"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import math
import cv2
from PIL import Image, ImageDraw, ImageFont
import pickle
import argparse
import os
from PIL import Image
import numpy as np
from argparse import Namespace
import torch
import importlib
import re
from matplotlib.lines import Line2D
# from util.coco import id2label
import matplotlib.pyplot as plt
import sys
# from util.layout import masks_to_layout
import json
sys.path.append("../util")

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
    eps = 1e-5
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + eps) * max(0, yB - yA + eps)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + eps) * (boxA[3] - boxA[1] + eps)
    boxBArea = (boxB[2] - boxB[0] + eps) * (boxB[3] - boxB[1] + eps)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate(
            [imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate(
            [imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().byte()
    # if label_tensor.size()[0] > 1:
    #    label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def tensor2attnmask(mask_tensor, n_label, imtype=np.uint8, tile=False):
    # n,m,h,w
    #if mask_tensor.dim() == 4:
    #    # transform each image in the batch
    #    images_np = []
    #    for b in range(mask_tensor.size(0)):
    #        one_image = mask_tensor[b]
    #        one_image_np = tensor2label(one_image, n_label, imtype)
    #        images_np.append(one_image_np.reshape(1, *one_image_np.shape))
    #    images_np = np.concatenate(images_np, axis=0)
    #    if tile:
    #        images_tiled = tile_images(images_np)
    #        return images_tiled
    #    else:
    #        images_np = images_np[0]
    #        return images_np
    assert mask_tensor.dim()==3
    mask_colorize = MaskColorize(n_label)
    colorized_tensor = mask_colorize(mask_tensor)
    colorized_tensor = torch.sqrt(colorized_tensor+1e-3)
    colorized_np = np.transpose(colorized_tensor.numpy(), (1, 2, 0))
    return colorized_np


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not isinstance(image_numpy, np.ndarray):
        image_pil = image_numpy
    else:
        if len(image_numpy.shape) == 2:
            image_numpy = np.expand_dims(image_numpy, axis=2)
        if image_numpy.shape[2] == 1:
            image_numpy = np.repeat(image_numpy, 3, 2)
        if image_numpy.shape[2]==3:
            mask = image_numpy.sum(axis=2,keepdims=True) > 1e-3
            alpha_mask = np.zeros_like(mask).astype(np.uint8)
            alpha_mask[mask]=255
            image_numpy = np.concatenate([image_numpy,alpha_mask],axis=2)
        image_pil = Image.fromarray(image_numpy,'RGBA')

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (
            module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230,
                                                                           150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153,
                                                                            153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220,
                                                                           20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n]).cuda()

    def __call__(self, label_tensor):
        size = label_tensor.size()
        color_image = torch.FloatTensor(3, size[1], size[2]).fill_(0).cuda()

        for label in range(0, len(self.cmap)):
            if label_tensor.size(0) == 1:
                mask = label_tensor[0] == label
                mask = mask.cuda()
            else:
                mask = label_tensor[label].cuda()
            color_image[0][mask] += self.cmap[label][0]
            color_image[1][mask] += self.cmap[label][1]
            color_image[2][mask] += self.cmap[label][2]
        color_image = color_image.byte()

        return color_image.cpu()


class MaskColorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n]).float()

    def __call__(self, label_tensor):
        label_tensor = label_tensor.cpu()
        size = label_tensor.size()
        color_image = torch.FloatTensor(3, size[1], size[2]).fill_(0)

        for label_idx in range(0, len(self.cmap)):
            mask = label_tensor[label_idx]
            color_image[0] += self.cmap[label_idx][0] * mask
            color_image[1] += self.cmap[label_idx][1] * mask
            color_image[2] += self.cmap[label_idx][2] * mask

        return color_image.cpu()


def plot_grad_flow_line(named_parameters, fig_name='grad_flow.jpg'):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
    plt.savefig(fig_name)


def plot_grad_flow_barchart(named_parameters, fig_name='grad_flow.jpg'):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
                max_grads.append(0)
                print(n)
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
    plt.figure(figsize=(8, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers,
               rotation="vertical", fontsize=8)
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.yscale('log')
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.subplots_adjust(bottom=0.4)
    print("Figure Generated.")
    plt.savefig(fig_name)


def drawrect(drawcontext, xy, outline=None, width=2):
    x1, y1, x2, y2 = xy
    x1 += width
    y1 += width
    x2 -= width
    y2 -= width
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    drawcontext.line(points, width=width, fill=outline)


color_names = ['red', 'blue', 'purple', 'green', 'black', 'orange', 'aqua', 'maroon', 'navy', 'teal', 'olive', 'gray', 'fuchsia', 'gold', 'red', 'yellow', 'blue', 'purple', 'green', 'black', 'orange',
               'aqua', 'maroon', 'navy', 'teal', 'olive', 'gray', 'fuchsia', 'gold', 'red', 'yellow', 'blue', 'purple', 'green', 'black', 'orange', 'aqua', 'maroon', 'navy', 'teal', 'olive', 'gray', 'fuchsia', 'gold']
font = ImageFont.truetype("arial.ttf", 15)


def box_has_intersect(bbox1, bbox2):
    return bb_intersection_over_union(bbox1, bbox2) > 1e-2


def cropped_to_full_mask(masks, bboxes):
    masks.unsqueeze(0)
    bboxes.unsqueeze(0)
    device = masks.get_device()
    O = masks.size(0)
    diag_emb = torch.zeros(O, O).to(device)
    diag_emb[torch.arange(O), torch.arange(O)] = 1
    obj2im = torch.zeros(O).to(device).long()
    full_mask = masks_to_layout(
        diag_emb, bboxes, masks, obj2im, H=256, N=1)
    full_mask.squeeze(0)
    return full_mask


def vis_keypoint(keypoints, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii
    return rendered image
    '''
    if format == 'coco':
        l_pair = [[16, 14], [14, 12], [17, 15],
                  [15, 13], [12, 13], [6, 12],
                  [7, 13], [6, 7], [6, 8], [7, 9],
                  [8, 10], [9, 11], [2, 3], [1, 2],
                  [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                  ]
        for pair in l_pair:
            pair[0] -= 1
            pair[1] -= 1
        # l_pair = [
        #    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        #    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        #    (17, 11), (17, 12),  # Body
        #    (11, 13), (12, 14), (13, 15), (14, 16)
        # ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191,
                                                                                    255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        p_color = p_color+p_color
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135,
                                                       255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        line_color = line_color+line_color
    else:
        raise NotImplementedError

    height = width = 256
    img = np.zeros((height, width, 3), dtype=np.uint8)
    keypoints[:, :, :2] *= 256
    for idx in range(keypoints.shape[0]):
        part_line = {}
        #kp_preds = keypoints[idx,:,:2]
        kp_preds = keypoints[idx, :, :2]
        kp_scores = keypoints[idx, :, 2]/2
        #kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        #kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(
                    length/2), int(stickwidth)), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                #cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(
                    0, min(1, 0.5*(kp_scores[start_p] + kp_scores[end_p])))

                img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
    #img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    alpha_mask = img.sum(axis=2) == 0
    alpha_ch = np.ones_like(alpha_mask).astype(np.uint8)*255
    alpha_ch[alpha_mask] = 0
    img_rgba[:, :, 3] = alpha_ch
    #cv2.imwrite("pose_im.png", img_rgba)
    return img_rgba


def optimal_position(current_bboxes, bbox_to_add, text_size, line_width):
    x1, y1, x2, y2 = bbox_to_add
    t_w, t_h = text_size
    start_x = x1-t_w
    start_y = y1-t_h
    end_x = max(x2+t_w,256)
    end_y = max(y2+t_h,256)
    cur_x = start_x
    cur_y = start_y

    def get_candidate_boxes():
        nonlocal cur_x
        nonlocal cur_y
        while cur_x < end_x:
            while cur_y < end_y:
                if cur_x > 0 and cur_y > 0 and cur_x+t_w < 256 and cur_y + t_h < 256:
                    candidate = [cur_x, cur_y, cur_x+t_w, cur_y+t_h]
                    yield candidate
                cur_y += t_h // 3
            cur_x += t_w // 3
            cur_y = start_y
    # four_candidate_captionbox = [
    #    [x1+line_width, y1+line_width, x1+line_width+t_w, y1+line_width+t_h],
    #    [x2-line_width-t_w, y1+line_width, x2-line_width, y1+line_width+t_w],
    #    [x1+line_width, y2-line_width-t_w, x1+line_width+t_w, y2-line_width],
    #    [x2-line_width-t_w, y2-line_width-t_h, x2-line_width, y2-line_width]
    # ]
    for caption_bbox in get_candidate_boxes():
        valid = True
        for cur_bbox in current_bboxes:
            if box_has_intersect(caption_bbox, cur_bbox):
                valid = False
                break
        if valid:
            return caption_bbox
    if len(current_bboxes) > 0:
        print("no solution found")
    return None
# with open("vg_obj_name_mapping.json") as infile:
#     vg_vocab_mapping = json.load(infile)
# rev_mapping = {val:key for key,val in vg_vocab_mapping.items()}
# rev_mapping[0]='person'

with open("/home/mark1123/data/mark1123/vg/vocab.json", 'r') as f:
    vocab = json.load(f)
    att_idx_to_name = vocab['attribute_idx_to_name']
    print(att_idx_to_name)
    object_idx_to_name = vocab['object_idx_to_name']


def draw_layout(bboxes, class_idxs, draw_caption, default_size=256, dset_mode='vcoco'):
    line_width = 2
    img_canvas = Image.new('RGBA', (default_size, default_size))
    draw_cxt = ImageDraw.Draw(img_canvas)
    num_objs = bboxes.shape[0]
    current_caption_boxes = []
    for i in range(num_objs):
        bbox = bboxes[i]
        bbox = [int(x) for x in bbox*default_size]
        #if class_idxs[i] == 1:
        if bbox[2]-bbox[0] < default_size*0.3 and bbox[3]-bbox[1] < default_size*0.3:
            current_caption_boxes.append(bbox)
        drawrect(draw_cxt, bbox, outline=color_names[i], width=line_width)
    for i in range(num_objs):
        bbox = bboxes[i]
        bbox = [int(x) for x in bbox*default_size]
        class_idx = class_idxs[i]
        if dset_mode == 'vcoco':
            text = id2label(class_idx)
        elif dset_mode == 'vgperson':
            text = object_idx_to_name[class_idx]
        text = object_idx_to_name[class_idx]
        text_size = font.getsize(text)
        text_bbox = Image.new('RGBA', text_size, color_names[i])
        if draw_caption:
            new_caption_bbox = optimal_position(current_caption_boxes,
                                                bbox, text_size, line_width)
            if new_caption_bbox is None:
                continue
            current_caption_boxes.append(new_caption_bbox)
            new_x, new_y, _, _ = list(map(int, new_caption_bbox))
            img_canvas.paste(text_bbox, (new_x, new_y))
            draw_cxt.text((new_x, new_y), text,
                          font=font, fill='white')
    # img_canvas.save("tmp.png")
    return img_canvas


def convert_sgdata(data_batch, opt, idx_mapping=None):
    imgs, objs, boxes, masks, obj_to_img = data_batch
    idx_mapping = {int(key): int(val) for key, val in idx_mapping.items()}
    bs = imgs.size(0)
    obj_select_mask = torch.zeros(objs.size(0), dtype=torch.bool)
    if idx_mapping is not None:
        for idx, obj in enumerate(objs):
            if int(obj) in idx_mapping:
                obj_select_mask[idx] = 1
        objs = objs[obj_select_mask]
        boxes = boxes[obj_select_mask]
        obj_to_img = obj_to_img[obj_select_mask]
    num_objs = torch.zeros(bs)
    for idx in range(bs):
        num_objs[idx] = torch.sum(obj_to_img == idx)
    bboxes = torch.zeros(bs, opt.max_len, 4)
    num_classes = torch.zeros(bs, opt.max_len)
    adj_mat = torch.zeros(bs, opt.max_len, opt.max_len)
    start = 0
    for idx in range(bs):
        bboxes[idx, :int(num_objs[idx]),
               :] = boxes[start:start+int(num_objs[idx])]
        num_classes[idx, :int(num_objs[idx])
                    ] = objs[start:start+int(num_objs[idx])] % 182
        start += int(num_objs[idx])
    for bs_idx in range(bs):
        for b1_idx, b1 in enumerate(bboxes[bs_idx]):
            for b2_idx, b2 in enumerate(bboxes[bs_idx]):
                adj_mat[bs_idx, b1_idx, b2_idx] = bb_intersection_over_union(
                    b1.numpy(), b2.numpy())
    data_i = {
        'num_objs': num_objs.long().unsqueeze(-1),
        'class_idx': num_classes.long(),
        'image': imgs,
        'bboxes': bboxes,
        'iou_adj': adj_mat
    }
    return data_i