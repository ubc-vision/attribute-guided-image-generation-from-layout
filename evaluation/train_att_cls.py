import sys
sys.path.insert(1, '/home/mark1123/layout2im')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.bilinear import crop_bbox_batch
from models.discriminator import AttributeDiscriminator128 as AttributeDiscriminator
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from utils.model_saver_iter import load_model, save_model
from pathlib import Path
from tensorboardX import SummaryWriter
from models.discriminator import add_sn


# attribute weight info

attribute_names = {'blonde': 0, 'tile': 1, 'white': 2, 'wooden': 3, 'light': 4, 'skateboarding': 5, 'walking': 6,
                   'dark': 7, 'red': 8, 'wet': 9, 'tiled': 10, 'grassy': 11, 'looking': 12, 'stuffed': 13, 'gold': 14,
                   'furry': 15, 'moving': 16, 'old': 17, 'wood': 18, 'distant': 19, 'framed': 20, 'colorful': 21,
                   'round': 22, 'fluffy': 23, 'tall': 24, 'brown': 25, 'orange': 26, 'watching': 27, 'silver': 28,
                   'running': 29, 'leafy': 30, 'purple': 31, 'covered': 32, 'circular': 33, 'dark brown': 34,
                   'sandy': 35, 'young': 36, 'laying': 37, 'skiing': 38, 'clear': 39, 'light blue': 40, 'dark blue': 41,
                   'standing': 42, 'spotted': 43, 'pink': 44, 'open': 45, 'paved': 46, 'short': 47, 'cloudy': 48,
                   'plaid': 49, 'chain link': 50, 'striped': 51, 'plastic': 52, 'glass': 53, 'hazy': 54, 'playing': 55,
                   'ceramic': 56, 'wearing': 57, 'gray': 58, 'dirty': 59, 'dirt': 60, 'beige': 61, 'large': 62,
                   'small': 63, "man's": 64, 'eating': 65, 'baby': 66, 'tan': 67, 'leafless': 68, 'parked': 69,
                   'yellow': 70, 'curly': 71, 'on': 72, 'jumping': 73, 'big': 74, 'khaki': 75, 'thick': 76, 'metal': 77,
                   'closed': 78, 'snowy': 79, 'sitting': 80, 'smiling': 81, 'dead': 82, 'rectangular': 83, 'long': 84,
                   'cement': 85, 'concrete': 86, 'surfing': 87, 'square': 88, 'clean': 89, 'green': 90, 'bright': 91,
                   'dry': 92, 'flying': 93, 'blue': 94, 'black': 95, 'light brown': 96, 'grazing': 97, 'cloudless': 98,
                   'bare': 99, 'brick': 100, 'overcast': 101, 'calm': 102, 'thin': 103, 'grey': 104, 'little': 105}

dic = {'white': 52795, 'black': 31290, 'green': 24967, 'blue': 24833, 'brown': 21859, 'red': 14886, 'large': 10613,
       'yellow': 8608, 'gray': 7811, 'wooden': 7511, 'grey': 6912, 'small': 6489, 'tall': 6140, 'long': 4801,
       'dark': 4698, 'standing': 4590, 'clear': 4530, 'metal': 4272, 'orange': 3707, 'tan': 3208, 'sitting': 2930,
       'silver': 2910, 'pink': 2839, 'wood': 2739, 'short': 2650, 'big': 2560, 'parked': 2442, 'brick': 2339,
       'cloudy': 2278, 'young': 2262, 'round': 2235, 'walking': 2221, 'striped': 2153, 'open': 2120, 'glass': 1996,
       'purple': 1647, 'bare': 1566, 'smiling': 1550, 'plastic': 1548, 'blonde': 1529, 'old': 1450, 'wet': 1322,
       'looking': 1307, 'bright': 1196, 'beige': 1134, 'concrete': 1059, 'playing': 1042, 'dirty': 995, 'calm': 993,
       'leafy': 977, 'light': 967, 'little': 940, 'eating': 929, 'colorful': 913, 'tiled': 872, 'grassy': 848,
       'square': 832, 'thick': 820, 'dry': 815, 'gold': 754, 'paved': 753, 'fluffy': 727, 'closed': 726, 'thin': 690,
       'light brown': 659, 'plaid': 647, 'dead': 647, 'light blue': 627, 'covered': 604, 'wearing': 599, 'sandy': 585,
       'cement': 552, 'dirt': 541, 'overcast': 539, 'leafless': 533, 'skiing': 526, 'distant': 525, 'flying': 515,
       'chain link': 513, 'watching': 494, 'grazing': 486, "man's": 482, 'running': 482, 'framed': 479, 'on': 478,
       'dark brown': 476, 'clean': 474, 'rectangular': 473, 'snowy': 460, 'cloudless': 444, 'furry': 442,
       'dark blue': 436, 'laying': 428, 'stuffed': 408, 'baby': 401, 'jumping': 396, 'moving': 391, 'spotted': 389,
       'tile': 382, 'curly': 379, 'hazy': 378, 'surfing': 373, 'skateboarding': 370, 'circular': 363, 'khaki': 362,
       'ceramic': 356}

weight = [dic[i] for i in attribute_names]
weight = [(100000 - i) / i for i in weight]  # 253468 is the total number of objects WITH GT labels

pos_weight = torch.tensor(weight)


def _downsample(x):
    return F.avg_pool2d(x, kernel_size=2)


class OptimizedBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.learnable_sc = (dim_in != dim_out) or downsample
        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        h = x
        if self.downsample:
            h = _downsample(x)
        return self.sc(h)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.learnable_sc = (dim_in != dim_out) or downsample

        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttributeDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, downsample_first=False, n_attribute = 128):
        super(AttributeDiscriminator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.main = nn.Sequential(
            # (3, 64, 64) -> (64, 64, 64)
            OptimizedBlock(3, conv_dim, downsample=downsample_first),
            # (64, 64, 64) -> (128, 32, 32)
            ResidualBlock(conv_dim, conv_dim * 2, downsample=True),
            # (128, 32, 32) -> (256, 16, 16)
            ResidualBlock(conv_dim * 2, conv_dim * 4, downsample=True),
            # (256, 16, 16) -> (512, 8, 8)
            ResidualBlock(conv_dim * 4, conv_dim * 8, downsample=True),
            # (512, 8, 8) -> (1024, 4, 4)
            ResidualBlock(conv_dim * 8, conv_dim * 16, downsample=True),
            # (1024, 4, 4) -> (1024, 2, 2)
            ResidualBlock(conv_dim * 16, conv_dim * 16, downsample=True),
        )

        # attribute
        self.classifier_att = nn.Linear(conv_dim * 16, n_attribute)
        self.sigmoid = nn.Sigmoid()

        # if n_class > 0:
        #     self.l_y = nn.Embedding(num_embeddings=n_class, embedding_dim=conv_dim * 16)

        # self.apply(weights_init)

    def forward(self, x):
        h = x
        h = self.main(h)
        h = self.relu(h)
        # (1024, 4, 4) -> (1024,)
        h = torch.sum(h, dim=(2, 3))

        # attribute
        att_cls = self.classifier_att(h)
        # att_cls = self.sigmoid(att_cls)

        return att_cls


def prepare_dir(name, path='~'):
    log_save_dir = '{}/checkpoints/att_cls_128/logs/{}'.format(path, name)
    model_save_dir = '{}/checkpoints/att_cls_128/models/{}'.format(path, name)
    sample_save_dir = '{}/checkpoints/att_cls_128/samples/{}'.format(path, name)
    result_save_dir = '{}/checkpoints/att_cls_128/results/{}'.format(path, name)

    if not Path(log_save_dir).exists(): Path(log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists(): Path(model_save_dir).mkdir(parents=True)
    if not Path(sample_save_dir).exists(): Path(sample_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)
    return log_save_dir, model_save_dir, sample_save_dir, result_save_dir


cudnn.benchmark = True
device = torch.device('cuda:0')
exp_name = 'training_att_cls_128'
log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(exp_name)

niter = 400000
attribute_nums = 106
batch_size = 12
save_step =2000
netD_att = AttributeDiscriminator(n_attribute=attribute_nums).to(device)
netD_att = add_sn(netD_att)
start_iter = load_model(netD_att, model_dir=model_save_dir, appendix='netD_attribute', iter='l')
netD_att_optimizer = torch.optim.Adam(netD_att.parameters(), 2e-4, [0.5, 0.999])

dataset = 'vg'
if dataset == 'vg':
    train_data_loader, val_data_loader = get_dataloader_vg(batch_size=batch_size,
                                                           attribute_embedding=attribute_nums)
elif dataset == 'coco':
    train_data_loader, val_data_loader = get_dataloader_coco(batch_size=batch_size)


data_loader = train_data_loader
data_iter = iter(data_loader)

if start_iter < niter:

    writer = SummaryWriter(log_save_dir)

    for i in range(start_iter, niter):

        try:
            batch = next(data_iter)
        except:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = batch
        imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = \
            imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), \
            obj_to_img, attribute.to(device), masks_shift.to(device), boxes_shift.to(device)

        crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, 64)

        att_cls = netD_att(crops_input.detach())
        att_idx = attribute.sum(dim=1).nonzero().squeeze()
        att_cls_annotated = torch.index_select(att_cls, 0, att_idx)
        attribute_annotated = torch.index_select(attribute, 0, att_idx)
        d_object_att_cls_loss_real = F.binary_cross_entropy_with_logits(att_cls_annotated, attribute_annotated,
                                                                        pos_weight=pos_weight.to(device))


        d_loss = 0
        d_loss += d_object_att_cls_loss_real

        netD_att.zero_grad()

        d_loss.backward()

        netD_att_optimizer.step()

        loss = {}
        loss['D/object_att_cls_loss'] = d_object_att_cls_loss_real.item()

        if (i + 1) % 100 == 0:
            log = 'iter [{:06d}/{:06d}]'.format(i + 1, niter)
            for tag, roi_value in loss.items():
                log += ", {}: {:.4f}".format(tag, roi_value)
            print(log)

        if (i + 1) % 500 == 0:
            for tag, roi_value in loss.items():
                writer.add_scalar(tag, roi_value, i + 1)

        if (i + 1) % save_step == 0:
            save_model(netD_att, model_dir=model_save_dir, appendix='netD_attribute', iter=i + 1, save_num=2,
                       save_step=save_step)

            # print("running evaluation at iter %d" % (i + 1))
            # import evaluation.gen_pickle_for_classification
            # import evaluation.compute_inception_score

    writer.close()