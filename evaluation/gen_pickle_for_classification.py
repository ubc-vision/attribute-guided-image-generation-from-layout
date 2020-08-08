import torch
from pathlib import Path
import sys

sys.path.insert(1, '/home/mark1123/layout2im')

# TODO: import 128 G, exp_name prepend 128
import argparse
from models.generator_obj_att128 import Generator
from models.generator_noclstm import Generator as Generator_noclstm
from models.discriminator_projection import ImageDiscriminator
from models.discriminator_projection import ObjectDiscriminator
from models.discriminator_projection import add_sn
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from data.coco_custom_mask import get_dataloader as get_dataloader_coco
from utils.model_saver_iter import load_model, save_model
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from data.utils import imagenet_deprocess_batch
from PIL import Image, ImageDraw
from imageio import imwrite
import os
import pickle
import train_att_change
from models.bilinear import crop_bbox_batch
from utils.draw_box import draw_layout


def str2bool(v):
    return v.lower() == 'true'


def draw_bbox_batch(images, bbox_sets):
    device = images.device
    results = []
    images = images.cpu().numpy()
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)), dtype=np.float32)
    for image, bbox_set in zip(images, bbox_sets):
        for bbox in bbox_set:
            if all(bbox == 0):
                continue
            else:
                image = draw_bbox(image, bbox)

        results.append(image)

    images = np.stack(results, axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).float().to(device)
    return images


def draw_bbox(image, bbox):
    im = Image.fromarray(np.uint8(image * 255))
    draw = ImageDraw.Draw(im)

    h, w, _ = image.shape
    c1 = (round(float(bbox[0] * w)), round(float(bbox[1] * h)))
    c2 = (round(float(bbox[2] * w)), round(float(bbox[3] * h)))

    draw.rectangle([c1, c2], outline=(0, 255, 0))

    output = np.array(im)/255

    return output


def prepare_dir(name,  path='~'):
    log_save_dir = '{}/checkpoints/all/logs/{}'.format(path, name)
    model_save_dir = '{}/checkpoints/all/models/{}'.format(path, name)
    sample_save_dir = '{}/checkpoints/all/samples/{}'.format(path, name)
    result_save_dir = '{}/checkpoints/all/results/final128_247_{}'.format(path, name)

    if not Path(log_save_dir).exists(): Path(log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists(): Path(model_save_dir).mkdir(parents=True)
    if not Path(sample_save_dir).exists(): Path(sample_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)
    return log_save_dir, model_save_dir, sample_save_dir, result_save_dir



def main(config):
    cudnn.benchmark = True
    device = torch.device('cuda:0')

    log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(config.exp_name)

    resinet_dir = "~/pickle/128_vg_pkl_resinet50_247k"

    if not os.path.exists(resinet_dir): os.mkdir(resinet_dir)

    attribute_nums =106
    if config.dataset == 'vg':
        train_data_loader, val_data_loader = get_dataloader_vg(batch_size=config.batch_size, attribute_embedding=attribute_nums)
    elif config.dataset == 'coco':
        train_data_loader, val_data_loader = get_dataloader_coco(batch_size=config.batch_size)
    vocab_num = train_data_loader.dataset.num_objects

    if config.clstm_layers == 0:
        netG = Generator_nolstm(num_embeddings=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim).to(device)
    else:
        netG = Generator(num_embeddings=vocab_num, obj_att_dim=config.embedding_dim, z_dim=config.z_dim,
                         clstm_layers=config.clstm_layers, obj_size=config.object_size,
                         attribute_dim=attribute_nums).to(device)

    _ = load_model(netG, model_dir=model_save_dir, appendix='netG', iter=config.resume_iter)

    netD_att = train_att_change.AttributeDiscriminator(n_attribute=attribute_nums).to(device)
    netD_att = train_att_change.add_sn(netD_att)

    start_iter_ = load_model(netD_att, model_dir="~/models/trained_models", appendix='netD_attribute', iter=config.resume_iter)
    _ = load_model(netG, model_dir=model_save_dir, appendix='netG', iter=config.resume_iter)

    data_loader = val_data_loader
    data_iter = iter(data_loader)

    with torch.no_grad():
        netG.eval()
        for i, batch in enumerate(data_iter):
            print('batch {}'.format(i))
            imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = batch
            att_idx = attribute.sum(dim=1).nonzero().squeeze()

            z = torch.randn(objs.size(0), config.z_dim)
            imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift = \
                imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), \
                obj_to_img, z.to(device), attribute.to(device), masks_shift.to(device), boxes_shift.to(device)

            # estimate attributes
            attribute_est = attribute.clone()
            att_mask = torch.zeros(attribute.shape[0])
            att_mask = att_mask.scatter(0, att_idx, 1).to(device)

            crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, config.object_size)
            estimated_att = netD_att(crops_input)
            max_idx = estimated_att.argmax(1)
            max_idx = max_idx.float() * (~att_mask.byte()).float().to(device)
            for row in range(attribute.shape[0]):
                if row not in att_idx:
                    attribute_est[row, int(max_idx[row])] = 1

            # Generate fake image
            output = netG(imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift, attribute_est)
            crops_input, crops_input_rec, crops_rand, crops_shift, img_rec, img_rand, img_shift, mu, logvar, z_rand_rec, z_rand_shift = output

            dict_to_save = {'imgs': imgs, 'imgs_rand': img_rand, 'imgs_shift': img_shift, 'objs': objs, 'boxes': boxes, 'boxes_shift': boxes_shift, 'obj_to_img': obj_to_img}

            out_name = os.path.join(resinet_dir, 'batch_{}.pkl'.format(i))
            pickle.dump(dict_to_save, open(out_name, 'wb'))

            img_rand = imagenet_deprocess_batch(img_rand)
            img_shift = imagenet_deprocess_batch(img_shift)

            # Save the generated images

            for j in range(img_rand.shape[0]):
                # layout = draw_layout(boxes[obj_to_img==j], objs[obj_to_img==j], True, dset_mode='vg')
                # img_path = os.path.join(result_save_dir, 'layout/img{:06d}_layout.png'.format(i*config.batch_size+j))
                # imwrite(img_path, layout)

                img_np = img_rand[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}.png'.format(i*config.batch_size+j))
                imwrite(img_path, img_np)

                # layout = draw_layout(boxes_shift[obj_to_img==j], objs[obj_to_img==j], True, dset_mode='vg')
                # img_path = os.path.join(result_save_dir, 'layout/img{:06d}_layouts.png'.format(i*config.batch_size+j))
                # imwrite(img_path, layout)

                img_np = img_shift[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}s.png'.format(i*config.batch_size+j))
                imwrite(img_path, img_np)

if True:
    # __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration
    path = '~'
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--dataset', type=str, default='vg')
    parser.add_argument('--vg_dir', type=str, default=path + '/vg')
    parser.add_argument('--coco_dir', type=str, default=path + '/coco')
    parser.add_argument('--batch_size', type=int, default=24)

    parser.add_argument('--niter', type=int, default=5000000, help='number of training iteration')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--object_size', type=int, default=64, help='image size')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--resi_num', type=int, default=6)
    parser.add_argument('--clstm_layers', type=int, default=3)

    # Loss weight
    parser.add_argument('--lambda_img_adv', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_obj_adv', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_obj_cls', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_z_rec', type=float, default=8.0, help='real/fake image')
    parser.add_argument('--lambda_img_rec', type=float, default=1.0, help='weight of reconstruction of image')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='real/fake image')

    # Log setting
    parser.add_argument('--resume_iter', type=str, default='l', help='l: from latest; s: from scratch; xxx: from iter xxx')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--tensorboard_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=500)
    parser.add_argument('--use_tensorboard', type=str2bool, default='true')

    # parser.add_argument('--exp_name', type=str, default='exp_e64z64')

    config = parser.parse_args()
    config.exp_name = '128_est_change_att_{}_bs{}e{}z{}clstm{}li{}lo{}lc{}lz{}lc{}lk{}'.format(config.dataset,
                                                                                          12,
                                                                                          config.embedding_dim,
                                                                                          config.z_dim,
                                                                                          config.clstm_layers,
                                                                                          config.lambda_img_adv,
                                                                                          config.lambda_obj_adv,
                                                                                          config.lambda_obj_cls,
                                                                                          config.lambda_z_rec,
                                                                                          config.lambda_img_rec,
                                                                                          config.lambda_kl)
    print(config)
    main(config)