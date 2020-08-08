import torch
from pathlib import Path
import argparse
from models.generator_obj_att import Generator
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from data.coco_custom_mask import get_dataloader as get_dataloader_coco
from utils.model_saver_iter import load_model, save_model
import torch.backends.cudnn as cudnn
import numpy as np
from data.utils import imagenet_deprocess_batch
from imageio import imwrite
import os
import train_att_change
from models.bilinear import crop_bbox_batch
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


colors = [(0, 255, 0),(0,0,0),(255,0,0),(0,0,255),(128,128,128),(255,96,208),(255,224,32),(0,192,0),(0,32,255),(255,208,160), (224, 224, 224)]


def str2bool(v):
    return v.lower() == 'true'


def draw_bbox_batch(images, bbox_sets, objs):
    device = images.device
    results = []
    images = images.cpu().numpy()
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)), dtype=np.float32)
    for image, bbox_set in zip(images, bbox_sets):
        for i, bbox in enumerate(bbox_set):
            if all(bbox == 0):
                continue
            else:
                try:
                    image = draw_bbox(image, bbox, i, objs)
                except:
                    continue
        results.append(image)

    images = np.stack(results, axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).float().to(device)
    return images


def draw_bbox(image, bbox, i, objs):
    im = Image.fromarray(np.uint8(image * 255))
    draw = ImageDraw.Draw(im)

    h, w, _ = image.shape
    c1 = (round(float(bbox[0] * w)), round(float(bbox[1] * h)))
    c2 = (round(float(bbox[2] * w)), round(float(bbox[3] * h)))

    draw.rectangle([c1, c2], outline=colors[i])
    draw.text((5, 5), "aa", font=ImageFont.truetype("arial"), fill=(255, 255, 0))

    output = np.array(im)/255

    return output


def prepare_dir(name,  path='~'):
    log_save_dir = '{}/checkpoints/all/logs/{}'.format(path, name)
    model_save_dir = '{}/checkpoints/all/models/{}'.format(path, name)
    sample_save_dir = '{}/checkpoints/all/samples/{}'.format(path, name)
    result_save_dir = '{}/checkpoints/all/results/{}'.format(path, name)

    if not Path(log_save_dir).exists(): Path(log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists(): Path(model_save_dir).mkdir(parents=True)
    if not Path(sample_save_dir).exists(): Path(sample_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)
    return log_save_dir, model_save_dir, sample_save_dir, result_save_dir



def main(config):
    cudnn.benchmark = True
    device = torch.device('cuda:0')

    log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(config.exp_name)

    resinet_dir = "~/pickle/vg_pkl_resinet50"

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

            # create boxes_set
            boxes_set = []
            for img in range(imgs.shape[0]):
                idx = list(torch.nonzero(obj_to_img == img).view(-1).numpy())
                boxes_set.append(boxes[idx])

            boxes_set_s = []
            for img in range(imgs.shape[0]):
                idx = list(torch.nonzero(obj_to_img == img).view(-1).numpy())
                boxes_set_s.append(boxes_shift[idx])

            img_rand = imagenet_deprocess_batch(img_rand)
            img_rand_box = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
            img_rand_box = draw_bbox_batch(img_rand_box, boxes_set, objs)

            img_shift = imagenet_deprocess_batch(img_shift)
            img_rand_s_box = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
            img_rand_s_box = draw_bbox_batch(img_rand_s_box, boxes_set_s, objs)

            # Save the generated images
            for j in range(img_rand.shape[0]):
                img_np = img_rand[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}.png'.format(i*config.batch_size+j))
                imwrite(img_path, img_np)

            for j in range(img_shift.shape[0]):
                img_np = img_rand_box[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_box.png'.format(i*config.batch_size+j))
                imwrite(img_path, img_np)

            for j in range(img_rand.shape[0]):
                img_np = img_shift[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_s.png'.format(i*config.batch_size+j))
                imwrite(img_path, img_np)

            for j in range(img_rand.shape[0]):
                img_np = img_rand_s_box[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_s_box.png'.format(i*config.batch_size+j))
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
    config.exp_name = 'est_change_att_{}_bs{}e{}z{}clstm{}li{}lo{}lc{}lz{}lc{}lk{}'.format(config.dataset,
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