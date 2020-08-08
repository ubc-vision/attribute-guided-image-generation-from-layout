#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL
import json
from torch.utils.data import DataLoader
from data.utils import imagenet_preprocess, Resize


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 normalize_images=True, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True, max_attributes_per_obj=30, attribute_embedding=128):
        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects - 1
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.max_attributes_per_obj = max_attributes_per_obj
        self.attribute_embedding = attribute_embedding

        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
        print(self.data['object_names'].shape)

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_paths[index])

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
            # obj_idxs = obj_idxs[0: self.max_objects]
        if len(obj_idxs) < self.max_objects and self.use_orphaned_objects:
            num_to_add = self.max_objects - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
            # obj_idxs += obj_idxs_without_rels[0: num_to_add]

        # random the obj_idxs
        random.shuffle(obj_idxs)

        O = len(obj_idxs)

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        masks = torch.zeros(O, 1, H, W)

        attributes = torch.zeros(O, self.max_attributes_per_obj, dtype=torch.long)
        one_hot_att = torch.zeros(O, self.attribute_embedding)

        boxes_shift = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        masks_shift = torch.zeros(O, 1, H, W)

        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            masks[i, :, round(y0 * H):round(y1 * H), round(x0 * W):round(x1 * W)] = 1
            attributes[i, :] = self.data['object_attributes'][index, obj_idx]

            # shift objs
            width = x1 - x0
            x0_shift = x0
            x1_shift = x1

            if width < 0.5:
                border_dist_left = x0
                border_dist_right = 1 - x1
                if border_dist_left > border_dist_right:
                    # shift = random.uniform(0, border_dist_left)
                    shift = border_dist_left * 0.8
                    x0_shift = x0 - shift
                    x1_shift = x1 - shift
                elif border_dist_right > border_dist_left:
                    # shift = random.uniform(0, border_dist_right)
                    shift = border_dist_right * 0.8
                    x0_shift = x0 + shift
                    x1_shift = x1 + shift
            masks_shift[i, :, round(y0 * H):round(y1 * H), round(x0_shift * W):round(x1_shift * W)] = 1
            boxes_shift[i] = torch.FloatTensor([x0_shift, y0, x1_shift, y1])

            # obj_idx_mapping[obj_idx] = i
            num_att = 0
            while attributes[i][num_att] != -1:
                num_att += 1
            # print(attributes[i, :], num_att)
            if num_att > 0:
                att = attributes[i, :].narrow(0, 0, num_att).unsqueeze_(0)
                # print(att)
                one_hot_att[i, :] = torch.zeros(1, self.attribute_embedding).scatter_(1, att, 1)
            else:
                one_hot_att[i, :] = torch.zeros(1, self.attribute_embedding)

        return image, objs, boxes, masks, one_hot_att, masks_shift, boxes_shift


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img, all_attributes, all_masks_shift, all_boxes_shift = [], [], [], [], [], [], [], []
    # obj_offset = 0
    for i, (img, objs, boxes, masks, attributes, masks_shift, boxes_shift) in enumerate(batch):
        all_imgs.append(img[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_attributes.append(attributes)
        all_masks_shift.append(masks_shift)
        all_boxes_shift.append(boxes_shift)
        # obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_attributes = torch.cat(all_attributes)

    # out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)

    all_masks_shift = torch.cat(all_masks_shift)
    all_boxes_shift = torch.cat(all_boxes_shift)
    # out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img, all_attributes)
    out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img, all_attributes, all_masks_shift, all_boxes_shift)

    return out


def get_dataloader(batch_size=10, VG_DIR='~/vg', VG_IMG_DIR="~/vg", attribute_embedding=128):
    vocab_json = os.path.join(VG_DIR, 'vocab.json')
    train_h5 = os.path.join(VG_DIR, 'train.h5')
    val_h5 = os.path.join(VG_DIR, 'test.h5')
    vg_image_dir = os.path.join(VG_IMG_DIR, 'images')
    image_size = (64, 64)
    num_train_samples = None
    max_objects_per_image = 10
    vg_use_orphaned_objects = True
    include_relationships = False
    batch_size = batch_size
    shuffle_val = False

    # build datasets
    with open(vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': train_h5,
        'image_dir': vg_image_dir,
        'image_size': image_size,
        'max_samples': num_train_samples,
        'max_objects': max_objects_per_image,
        'use_orphaned_objects': vg_use_orphaned_objects,
        'include_relationships': include_relationships,
        'attribute_embedding': attribute_embedding,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': False,
        'collate_fn': vg_collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = shuffle_val
    loader_kwargs['num_workers'] = 1
    val_loader = DataLoader(val_dset, **loader_kwargs)

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(batch_size=32, VG_DIR='/scratch/markma/sg2im/sg2im/data', VG_IMG_DIR ="/scratch/zhaobo/Datasets/vg", attribute_embedding=106)

    for i, batch in enumerate(train_loader):
        imgs, objs, boxes, masks, obj_to_img, atts, masks_shift, boxes_shift = batch

        print(atts.shape)
