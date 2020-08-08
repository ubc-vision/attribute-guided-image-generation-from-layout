import torch
from pathlib import Path
import argparse
from models.generator_obj_att128 import Generator
from models.discriminator import ImageDiscriminator
from models.discriminator import ObjectDiscriminator
from models.discriminator import AttributeDiscriminator
from models.discriminator import add_sn
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from utils.model_saver_iter import load_model, save_model
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from data.utils import imagenet_deprocess_batch
from PIL import Image, ImageDraw
from models.bilinear import crop_bbox_batch

import math
import random
from attribute_names import attribute_names
from attribute_counts import attribute_counts

# attribute weight info
weight = [attribute_counts[i] for i in attribute_names]
weight = [(100000 - i) / i for i in weight]  # 253468 is the total number of objects WITH GT labels

pos_weight = torch.tensor(weight)

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

    output = np.array(im) / 255

    return output


def prepare_dir(name, path='~'):
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
    matrix = torch.load("matrix_obj_vs_att.pt")
    cudnn.benchmark = True
    device = torch.device('cuda:1')

    log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(config.exp_name)

    attribute_nums = 106

    data_loader, _ = get_dataloader_vg(batch_size=config.batch_size, attribute_embedding=attribute_nums)

    vocab_num = data_loader.dataset.num_objects

    netG = Generator(num_embeddings=vocab_num, obj_att_dim=config.embedding_dim, z_dim=config.z_dim,
                         clstm_layers=config.clstm_layers, obj_size=config.object_size,
                         attribute_dim=attribute_nums).to(device)

    netD_image = ImageDiscriminator(conv_dim=config.embedding_dim).to(device)
    netD_object = ObjectDiscriminator(n_class=vocab_num).to(device)
    netD_att = AttributeDiscriminator(n_attribute=attribute_nums).to(device)

    netD_image = add_sn(netD_image)
    netD_object = add_sn(netD_object)
    netD_att = add_sn(netD_att)

    netG_optimizer = torch.optim.Adam(netG.parameters(), config.learning_rate, [0.5, 0.999])
    netD_image_optimizer = torch.optim.Adam(netD_image.parameters(), config.learning_rate, [0.5, 0.999])
    netD_object_optimizer = torch.optim.Adam(netD_object.parameters(), config.learning_rate, [0.5, 0.999])
    netD_att_optimizer = torch.optim.Adam(netD_att.parameters(), config.learning_rate, [0.5, 0.999])

    start_iter_ = load_model(netD_object, model_dir=model_save_dir, appendix='netD_object', iter=config.resume_iter)

    start_iter_ = load_model(netD_att, model_dir=model_save_dir, appendix='netD_attribute', iter=config.resume_iter)

    start_iter_ = load_model(netD_image, model_dir=model_save_dir, appendix='netD_image', iter=config.resume_iter)

    start_iter = load_model(netG, model_dir=model_save_dir, appendix='netG', iter=config.resume_iter)

    data_iter = iter(data_loader)

    if start_iter < config.niter:

        if config.use_tensorboard: writer = SummaryWriter(log_save_dir)

        for i in range(start_iter, config.niter):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = batch
            z = torch.randn(objs.size(0), config.z_dim)

            att_idx = attribute.sum(dim=1).nonzero().squeeze()
            # print("Train D")
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift \
                = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img, z.to(
                device), attribute.to(device), masks_shift.to(device), boxes_shift.to(device)

            attribute_GT = attribute.clone()

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


            # change GT attribute:
            num_img_to_change = math.floor(imgs.shape[0]/3)
            for img_idx in range(num_img_to_change):
                obj_indices = torch.nonzero(obj_to_img == img_idx).view(-1)

                num_objs_to_change = math.floor(len(obj_indices)/2)
                for changed, obj_idx in enumerate(obj_indices):
                    if changed >= num_objs_to_change:
                        break
                    obj = objs[obj_idx]
                    # change GT attribute
                    old_attributes = torch.nonzero(attribute_GT[obj_idx]).view(-1)
                    new_attribute = random.choices(range(106), matrix[obj].scatter(0, old_attributes.cpu(), 0),
                                                   k=random.randrange(1, 3))
                    attribute[obj_idx] = 0  # remove all attributes for obj
                    attribute[obj_idx] = attribute[obj_idx].scatter(0, torch.LongTensor(new_attribute).to(device), 1)   # assign new attribute

                    # change estimated attributes
                    attribute_est[obj_idx] = 0  # remove all attributes for obj
                    attribute_est[obj_idx] = attribute[obj_idx].scatter(0, torch.LongTensor(new_attribute).to(device), 1)

            # Generate fake image
            output = netG(imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift, attribute_est)
            crops_input, crops_input_rec, crops_rand, crops_shift, img_rec, img_rand, img_shift, mu, logvar, z_rand_rec, z_rand_shift = output

            # Compute image adv loss with fake images.
            out_logits = netD_image(img_rec.detach())
            d_image_adv_loss_fake_rec = F.binary_cross_entropy_with_logits(out_logits,
                                                                           torch.full_like(out_logits, 0))

            out_logits = netD_image(img_rand.detach())
            d_image_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits,
                                                                            torch.full_like(out_logits, 0))

            # shift image adv loss
            out_logits = netD_image(img_shift.detach())
            d_image_adv_loss_fake_shift = F.binary_cross_entropy_with_logits(out_logits,
                                                                             torch.full_like(out_logits, 0))

            d_image_adv_loss_fake = 0.4 * d_image_adv_loss_fake_rec + 0.4 * d_image_adv_loss_fake_rand + 0.2 * d_image_adv_loss_fake_shift

            # Compute image src loss with real images rec.
            out_logits = netD_image(imgs)
            d_image_adv_loss_real = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 1))

            # Compute object sn adv loss with fake rec crops
            out_logits, _ = netD_object(crops_input_rec.detach(), objs)
            g_object_adv_loss_rec = F.binary_cross_entropy_with_logits(out_logits, torch.full_like(out_logits, 0))

            # Compute object sn adv loss with fake rand crops
            out_logits, _ = netD_object(crops_rand.detach(), objs)

            d_object_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits,
                                                                             torch.full_like(out_logits, 0))

            # shift obj adv loss
            out_logits, _ = netD_object(crops_shift.detach(), objs)
            d_object_adv_loss_fake_shift = F.binary_cross_entropy_with_logits(out_logits,
                                                                              torch.full_like(out_logits, 0))

            d_object_adv_loss_fake = 0.4 * g_object_adv_loss_rec + 0.4 * d_object_adv_loss_fake_rand + 0.2 * d_object_adv_loss_fake_shift

            # Compute object sn adv loss with real crops.
            out_logits_src, out_logits_cls = netD_object(crops_input.detach(), objs)

            d_object_adv_loss_real = F.binary_cross_entropy_with_logits(out_logits_src,
                                                                        torch.full_like(out_logits_src, 1))

            # cls
            d_object_cls_loss_real = F.cross_entropy(out_logits_cls, objs)
            # attribute
            att_cls = netD_att(crops_input.detach())
            att_idx = attribute_GT.sum(dim=1).nonzero().squeeze()
            att_cls_annotated = torch.index_select(att_cls, 0, att_idx)
            attribute_annotated = torch.index_select(attribute_GT, 0, att_idx)
            d_object_att_cls_loss_real = F.binary_cross_entropy_with_logits(att_cls_annotated, attribute_annotated,
                                                                            pos_weight=pos_weight.to(device))

            # Backward and optimize.
            d_loss = 0
            d_loss += config.lambda_img_adv * (d_image_adv_loss_fake + d_image_adv_loss_real)
            d_loss += config.lambda_obj_adv * (d_object_adv_loss_fake + d_object_adv_loss_real)
            d_loss += config.lambda_obj_cls * d_object_cls_loss_real
            d_loss += config.lambda_att_cls * d_object_att_cls_loss_real

            netD_image.zero_grad()
            netD_object.zero_grad()
            netD_att.zero_grad()

            d_loss.backward()

            netD_image_optimizer.step()
            netD_object_optimizer.step()
            netD_att_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss'] = d_loss.item()
            loss['D/image_adv_loss_real'] = d_image_adv_loss_real.item()
            loss['D/image_adv_loss_fake'] = d_image_adv_loss_fake.item()
            loss['D/object_adv_loss_real'] = d_object_adv_loss_real.item()
            loss['D/object_adv_loss_fake'] = d_object_adv_loss_fake.item()
            loss['D/object_cls_loss_real'] = d_object_cls_loss_real.item()
            loss['D/object_att_cls_loss'] = d_object_att_cls_loss_real.item()

     # print("train G")
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # Generate fake image

            output = netG(imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift, attribute_est)
            crops_input, crops_input_rec, crops_rand, crops_shift, img_rec, img_rand, img_shift, mu, logvar, z_rand_rec, z_rand_shift = output

            # reconstruction loss of ae and img
            rec_img_mask = torch.ones(imgs.shape[0]).scatter(0, torch.LongTensor(range(num_img_to_change)), 0).to(
                device)
            g_img_rec_loss = rec_img_mask * torch.abs(img_rec - imgs).view(imgs.shape[0], -1).mean(1)
            g_img_rec_loss = g_img_rec_loss.sum() / (imgs.shape[0] - num_img_to_change)

            g_z_rec_loss_rand = torch.abs(z_rand_rec - z).mean()
            g_z_rec_loss_shift = torch.abs(z_rand_shift - z).mean()
            g_z_rec_loss = 0.5 * g_z_rec_loss_rand + 0.5 * g_z_rec_loss_shift

            # kl loss
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            g_kl_loss = torch.sum(kl_element).mul_(-0.5)

            # Compute image adv loss with fake images.
            out_logits = netD_image(img_rec)

            g_image_adv_loss_fake_rec = F.binary_cross_entropy_with_logits(out_logits,
                                                                           torch.full_like(out_logits, 1))

            out_logits = netD_image(img_rand)
            g_image_adv_loss_fake_rand = F.binary_cross_entropy_with_logits(out_logits,
                                                                            torch.full_like(out_logits, 1))

            # shift image adv loss
            out_logits = netD_image(img_shift)
            g_image_adv_loss_fake_shift = F.binary_cross_entropy_with_logits(out_logits,
                                                                             torch.full_like(out_logits, 1))


            g_image_adv_loss_fake = 0.4 * g_image_adv_loss_fake_rec + 0.4 * g_image_adv_loss_fake_rand + 0.2 * g_image_adv_loss_fake_shift

            # Compute object adv loss with fake images.
            out_logits_src, out_logits_cls = netD_object(crops_input_rec, objs)

            g_object_adv_loss_rec = F.binary_cross_entropy_with_logits(out_logits_src,
                                                                       torch.full_like(out_logits_src, 1))
            g_object_cls_loss_rec = F.cross_entropy(out_logits_cls, objs)
            # attribute
            att_cls = netD_att(crops_input_rec)
            att_idx = attribute.sum(dim=1).nonzero().squeeze()
            attribute_annotated = torch.index_select(attribute, 0, att_idx)
            att_cls_annotated = torch.index_select(att_cls, 0, att_idx)
            g_object_att_cls_loss_rec = F.binary_cross_entropy_with_logits(att_cls_annotated, attribute_annotated,
                                                                           pos_weight=pos_weight.to(device))

            out_logits_src, out_logits_cls = netD_object(crops_rand, objs)
            g_object_adv_loss_rand = F.binary_cross_entropy_with_logits(out_logits_src,
                                                                        torch.full_like(out_logits_src, 1))
            g_object_cls_loss_rand = F.cross_entropy(out_logits_cls, objs)
            # attribute
            att_cls = netD_att(crops_rand)
            att_cls_annotated = torch.index_select(att_cls, 0, att_idx)
            g_object_att_cls_loss_rand = F.binary_cross_entropy_with_logits(att_cls_annotated, attribute_annotated,
                                                                            pos_weight=pos_weight.to(device))

            # shift adv obj loss
            out_logits_src, out_logits_cls = netD_object(crops_shift, objs)
            g_object_adv_loss_shift = F.binary_cross_entropy_with_logits(out_logits_src,
                                                                         torch.full_like(out_logits_src, 1))

            g_object_cls_loss_shift = F.cross_entropy(out_logits_cls, objs)
            # attribute
            att_cls = netD_att(crops_shift)
            att_cls_annotated = torch.index_select(att_cls, 0, att_idx)
            g_object_att_cls_loss_shift = F.binary_cross_entropy_with_logits(att_cls_annotated, attribute_annotated,
                                                                             pos_weight=pos_weight.to(device))

            g_object_att_cls_loss = 0.4 * g_object_att_cls_loss_rec + 0.4 * g_object_att_cls_loss_rand + 0.2 * g_object_att_cls_loss_shift

            g_object_adv_loss = 0.4 * g_object_adv_loss_rec + 0.4 * g_object_adv_loss_rand + 0.2 * g_object_adv_loss_shift
            g_object_cls_loss = 0.4 * g_object_cls_loss_rec + 0.4 * g_object_cls_loss_rand + 0.2 * g_object_cls_loss_shift

            # Backward and optimize.
            g_loss = 0
            g_loss += config.lambda_img_rec * g_img_rec_loss
            g_loss += config.lambda_z_rec * g_z_rec_loss
            g_loss += config.lambda_img_adv * g_image_adv_loss_fake
            g_loss += config.lambda_obj_adv * g_object_adv_loss
            g_loss += config.lambda_obj_cls * g_object_cls_loss
            g_loss += config.lambda_att_cls * g_object_att_cls_loss
            g_loss += config.lambda_kl * g_kl_loss

            netG.zero_grad()

            g_loss.backward()

            netG_optimizer.step()

            loss['G/loss'] = g_loss.item()
            loss['G/image_adv_loss'] = g_image_adv_loss_fake.item()
            loss['G/object_adv_loss'] = g_object_adv_loss.item()
            loss['G/object_cls_loss'] = g_object_cls_loss.item()
            loss['G/rec_img'] = g_img_rec_loss.item()
            loss['G/rec_z'] = g_z_rec_loss.item()
            loss['G/kl'] = g_kl_loss.item()
            loss['G/object_att_cls_loss'] = g_object_att_cls_loss.item()

            # =================================================================================== #
            #                               4. Log                                                #
            # =================================================================================== #
            if (i + 1) % config.log_step == 0:
                log = 'iter [{:06d}/{:06d}]'.format(i + 1, config.niter)
                for tag, roi_value in loss.items():
                    log += ", {}: {:.4f}".format(tag, roi_value)
                print(log)

            if (i + 1) % config.tensorboard_step == 0 and config.use_tensorboard:
                for tag, roi_value in loss.items():

                    writer.add_scalar(tag, roi_value, i + 1)
                writer.add_images('Result/crop_real', imagenet_deprocess_batch(crops_input).float() / 255, i + 1)
                writer.add_images('Result/crop_real_rec', imagenet_deprocess_batch(crops_input_rec).float() / 255,
                                  i + 1)
                writer.add_images('Result/crop_rand', imagenet_deprocess_batch(crops_rand).float() / 255, i + 1)
                writer.add_images('Result/img_real', imagenet_deprocess_batch(imgs).float() / 255, i + 1)
                writer.add_images('Result/img_real_rec', imagenet_deprocess_batch(img_rec).float() / 255,
                                  i + 1)
                writer.add_images('Result/img_fake_rand', imagenet_deprocess_batch(img_rand).float() / 255,
                                  i + 1)

            if (i + 1) % config.save_step == 0:

                # netG_noDP.load_state_dict(new_state_dict)
                save_model(netG, model_dir=model_save_dir, appendix='netG', iter=i + 1, save_num=2,
                           save_step=config.save_step)
                save_model(netD_image, model_dir=model_save_dir, appendix='netD_image', iter=i + 1, save_num=2,
                           save_step=config.save_step)
                save_model(netD_object, model_dir=model_save_dir, appendix='netD_object', iter=i + 1, save_num=2,
                           save_step=config.save_step)
                save_model(netD_att, model_dir=model_save_dir, appendix='netD_attribute', iter=i + 1, save_num=2,
                           save_step=config.save_step)

        if config.use_tensorboard: writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration
    path = '~'
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--dataset', type=str, default='vg')
    parser.add_argument('--vg_dir', type=str, default=path + '/vg')
    parser.add_argument('--coco_dir', type=str, default=path + '/coco')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--niter', type=int, default=900000, help='number of training iteration')

    parser.add_argument('--image_size', type=int, default=128, help='image size')
    parser.add_argument('--object_size', type=int, default=64, help='image size')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--resi_num', type=int, default=6)
    parser.add_argument('--clstm_layers', type=int, default=3)

    # Loss weight
    parser.add_argument('--lambda_img_adv', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_obj_adv', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_obj_cls', type=float, default=1.0, help='real/fake image')
    parser.add_argument('--lambda_z_rec', type=float, default=8.0, help='real/fake image')
    parser.add_argument('--lambda_img_rec', type=float, default=1.0, help='weight of reconstruction of image')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='real/fake image')
    # attribute
    parser.add_argument('--lambda_att_cls', type=float, default=2.0, help='real/fake image')

    # Log setting
    parser.add_argument('--resume_iter', type=str, default='l',
                        help='l: from latest; s: from scratch; xxx: from iter xxx')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--tensorboard_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=500)
    parser.add_argument('--use_tensorboard', type=str2bool, default='true')

    config = parser.parse_args()
    config.exp_name = 'est_change_att_{}_bs{}e{}z{}clstm{}li{}lo{}lc{}lz{}lc{}lk{}'.format(config.dataset,
                                                                                              config.batch_size,
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
