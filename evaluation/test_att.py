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

import argparse, json, os

import numpy as np
from sklearn.metrics import confusion_matrix
from imageio import imwrite
import torch
import torch.nn as nn
import torch.nn.functional as F


VG_DIR = os.path.expanduser('data')
parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--checkpoint', default='checkpoint_with_model.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs2')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='128,128', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

parser.add_argument('--netD_save_dir', default='~/models/trained_models/')


def safe_division(n, d):
    return n / d if d else 1  # return 1 when predict nothing

def build_vg_dsets(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': args.train_h5,
        'image_dir': args.vg_image_dir,
        'image_size': args.image_size,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.vg_use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_loaders(args):
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def main(args):
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if not os.path.isdir(args.output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    model = Sg2ImModel(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)

    attribute_num = 106

    vocab, train_loader, val_loader = build_loaders(args)
    data_iter = iter(val_loader)

    netD_attribute = AttributeDiscriminator(n_attribute=attribute_num).cuda()
    netD_attribute = add_sn(netD_attribute)

    # start_iter_ = load_model(netD_object, model_dir=model_save_dir, appendix='netD_object', iter=config.resume_iter)
    _ = load_model(netD_attribute, model_dir=args.netD_save_dir, appendix='netD_attribute',
                   iter='l')

    attributes_pred = np.zeros((253468, attribute_num))
    attributes_gt = None

    cur_obj_start_idx = 0
    for b, batch in enumerate(data_iter):
        print('batch {}'.format(b))

        batch = [tensor.cuda() for tensor in batch]
        masks = None
        if len(batch) == 7:  # vg
            imgs, objs, boxes, triples, obj_to_img, triple_to_img, attribute_orig = batch
        elif len(batch) == 8:  # coco
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, dummy = batch

        # Load the scene graphs
        # with open(args.scene_graphs_json, 'r') as f:
        #   scene_graphs = json.load(f)

        # Run the model forward
        with torch.no_grad():
            imgs_rand, boxes_pred, masks_pred, _ = model(objs, triples, obj_to_img, attribute_orig)

        crops_rand = crop_bbox_batch(imgs_rand, boxes, obj_to_img, 32)

        att_idx = attribute_orig.sum(dim=1).nonzero().squeeze()

        crops_input = torch.index_select(crops_rand, 0, att_idx)
        attribute = torch.index_select(attribute_orig, 0, att_idx)

        att_cls = netD_attribute(crops_input)
        att_cls = F.sigmoid(att_cls)

        for i in att_idx:

            ll = att_idx.cpu().numpy().tolist().index(i)

            non0_idx_cls = (att_cls[ll] > 0.9).nonzero()

            pred_idx = non0_idx_cls.squeeze().cpu().numpy()

            attributes_pred[cur_obj_start_idx, pred_idx] = 1

            cur_obj_start_idx += 1

        # construct GT array
        attributes_gt = attribute.cpu() if attributes_gt is None else np.vstack([attributes_gt, attribute.cpu()])

        # Save the generated images
        imgs_rand = imagenet_deprocess_batch(imgs_rand)
        for i in range(imgs_rand.shape[0]):
            img_np = imgs_rand[i].numpy().transpose(1, 2, 0)
            img_path = os.path.join(args.output_dir, 'img%06d.png' % (b * args.batch_size + i))
            imwrite(img_path, img_np)

        # Draw the scene graphs
        if args.draw_scene_graphs == 1:
            for i, sg in enumerate(scene_graphs):
                sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
                sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % (b * args.batch_size + i))
                imwrite(sg_img_path, sg_img)

    # calculate recall precision
    num_data = attributes_gt.shape[0]
    count = np.zeros((num_data, 4))
    recall, precision = np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        # tn, fp, fn, tp = confusion_matrix(attributes_pred, attributes_gt).ravel()
        count[i] = confusion_matrix(attributes_gt[i], attributes_pred[i]).ravel()
        recall[i] = count[i][3] / (count[i][3] + count[i][2])
        precision[i] = safe_division(count[i][3], count[i][3] + count[i][1])

    print("average precision = {}".format(precision.mean()))
    print("average recall = {}".format(recall.mean()))

    print("average pred # per obj")
    print((count[:, 1].sum() + count[:, 3].sum()) / count.shape[0])

    print("average GT # per obj")
    print((count[:, 2].sum() + count[:, 3].sum()) / count.shape[0])

    print("% of data that predict something")
    print(((count[:, 1] + count[:, 3]) > 0).sum() / count.shape[0])

    print("% of data at least predicted correct once")
    print((count[:, 3] > 0).sum() / count.shape[0])    # calculate recall precision
    num_data = attributes_gt.shape[0]
    count = np.zeros((num_data, 4))
    recall, precision = np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        # tn, fp, fn, tp = confusion_matrix(attributes_pred, attributes_gt).ravel()
        count[i] = confusion_matrix(attributes_gt[i], attributes_pred[i]).ravel()
        recall[i] = count[i][3] / (count[i][3] + count[i][2])
        precision[i] = safe_division(count[i][3], count[i][3] + count[i][1])

    print("average precision = {}".format(precision.mean()))
    print("average recall = {}".format(recall.mean()))

    print("average pred # per obj")
    print((count[:, 1].sum() + count[:, 3].sum()) / count.shape[0])

    print("average GT # per obj")
    print((count[:, 2].sum() + count[:, 3].sum()) / count.shape[0])

    print("% of data that predict something")
    print(((count[:, 1] + count[:, 3]) > 0).sum() / count.shape[0])

    print("% of data at least predicted correct once")
    print((count[:, 3] > 0).sum() / count.shape[0])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
