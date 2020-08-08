import torch
from pathlib import Path
import argparse
from models.generator_obj_att128 import Generator
from models.discriminator import AttributeDiscriminator128 as AttributeDiscriminator
from models.discriminator import add_sn
from data.vg_custom_mask import get_dataloader as get_dataloader_vg
from utils.model_saver_iter import load_model
import torch.backends.cudnn as cudnn
import numpy as np
from data.utils import imagenet_deprocess_batch
from PIL import Image, ImageDraw
from imageio import imwrite
import os
import json
from models.bilinear import crop_bbox_batch
from sklearn.metrics import confusion_matrix

def safe_division(n, d):
    return n / d if d else 1  # return 1 when predict nothing


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
    result_save_dir = '{}/checkpoints/all/results/{}'.format(path, name)

    if not Path(log_save_dir).exists(): Path(log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists(): Path(model_save_dir).mkdir(parents=True)
    if not Path(sample_save_dir).exists(): Path(sample_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists(): Path(result_save_dir).mkdir(parents=True)
    return log_save_dir, model_save_dir, sample_save_dir, result_save_dir


def main(config):
    cudnn.benchmark = True

    device = torch.device('cuda:1')

    log_save_dir, model_save_dir, sample_save_dir, result_save_dir = prepare_dir(config.exp_name)

    with open("~/vg/vocab.json", 'r') as f:
        vocab = json.load(f)
        att_idx_to_name = np.array(vocab['attribute_idx_to_name'])
        print(att_idx_to_name)
        object_idx_to_name = np.array(vocab['object_idx_to_name'])


    attribute_nums =106

    train_data_loader, val_data_loader = get_dataloader_vg(batch_size=config.batch_size, attribute_embedding=attribute_nums)

    vocab_num = train_data_loader.dataset.num_objects

    netG = Generator(num_embeddings=vocab_num, obj_att_dim=config.embedding_dim, z_dim=config.z_dim,
                         clstm_layers=config.clstm_layers, obj_size=config.object_size,
                         attribute_dim=attribute_nums).to(device)

    netD_att = AttributeDiscriminator(n_attribute=attribute_nums).to(device)
    netD_att = add_sn(netD_att)

    start_iter_ = load_model(netD_att, model_dir="~/models/trained_models", appendix='netD_attribute', iter=config.resume_iter)
    _ = load_model(netG, model_dir=model_save_dir, appendix='netG', iter=config.resume_iter)

    data_loader = val_data_loader
    data_iter = iter(data_loader)

    cur_obj_start_idx = 0
    attributes_pred = np.zeros((253468, attribute_nums))
    attributes_gt = None

    with torch.no_grad():
        netG.eval()
        for i, batch in enumerate(data_iter):
            print('batch {}'.format(i))

            imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = batch
            z = torch.randn(objs.size(0), config.z_dim)
            att_idx = attribute.sum(dim=1).nonzero().squeeze()

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

            # predict attribute on generated
            crops_rand_yes = torch.index_select(crops_rand, 0, att_idx.to(device))
            attribute_yes = torch.index_select(attribute, 0, att_idx.to(device))

            estimated_att_rand = netD_att(crops_rand_yes)
            att_cls = torch.sigmoid(estimated_att_rand)

            for k in att_idx:
                ll = att_idx.cpu().numpy().tolist().index(k)

                non0_idx_cls = (att_cls[ll] > 0.9).nonzero()

                pred_idx = non0_idx_cls.squeeze().cpu().numpy()

                attributes_pred[cur_obj_start_idx, pred_idx] = 1

                cur_obj_start_idx += 1

            # construct GT array
            attributes_gt = attribute_yes.clone().cpu() if attributes_gt is None else np.vstack(
                [attributes_gt, attribute_yes.clone().cpu()])

            img_rand = imagenet_deprocess_batch(img_rand)
            img_shift = imagenet_deprocess_batch(img_shift)
            img_rec = imagenet_deprocess_batch(img_rec)

            # attribute modification
            changed_list = []
            src = 2     # 94 blue, 95 black
            tgt = 95     # 8 red, 2 white
            for idx, o in enumerate(objs):
                    attribute[idx, [2, 8, 0, 94, 90, 95, 96, 34, 25, 70, 58, 104]] = 0 # remove other color
                    attribute[idx, tgt] = 1

                    attribute_est[idx, [2, 8, 0, 94, 90, 95, 96, 34, 25, 70, 58, 104]] = 0 # remove other color
                    attribute_est[idx, tgt] = 1
                    changed_list.append(idx)

            # Generate red image
            z = torch.randn(objs.size(0), config.z_dim).to(device)
            output = netG(imgs, objs, boxes, masks, obj_to_img, z, attribute, masks_shift, boxes_shift, attribute_est)
            crops_input, crops_input_rec, crops_rand_y, crops_shift_y, img_rec_y, img_rand_y, img_shift_y, mu, logvar, z_rand_rec, z_rand_shift = output

            img_rand_y = imagenet_deprocess_batch(img_rand_y)
            img_shift_y = imagenet_deprocess_batch(img_shift_y)
            img_rec_y = imagenet_deprocess_batch(img_rec_y)
            imgs = imagenet_deprocess_batch(imgs)

            #2 top k
            estimated_att = netD_att(crops_rand)
            max_idx = estimated_att.topk(5)[1]
            changed_list = [i for i in changed_list if tgt not in max_idx[i]]
            estimated_att_y = netD_att(crops_rand_y)
            max_idx_y = estimated_att_y.topk(3)[1]
            changed_list_success = [i for i in changed_list if tgt in max_idx_y[i]]
            for j in range(imgs.shape[0]):

                img_np = img_shift[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_shift.png'.format(i * config.batch_size + j))
                imwrite(img_path, img_np)

                img_np = img_rand[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_rand.png'.format(i * config.batch_size + j))
                imwrite(img_path, img_np)

                img_rec_np = img_rec[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir,
                                        'img{:06d}_rec.png'.format(i * config.batch_size + j))
                imwrite(img_path, img_rec_np)

                img_real_np = imgs[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(result_save_dir, 'img{:06d}_real.png'.format(i * config.batch_size + j))
                imwrite(img_path, img_real_np)

                cur_obj_success = [int(objs[c]) for c in changed_list_success if obj_to_img[c] == j]

                # save successfully modified images
                if len(cur_obj_success) > 0:

                    img_shift_y = img_shift_y[j].numpy().transpose(1, 2, 0)
                    img_path_red = os.path.join(result_save_dir, 'img{:06d}_shift_{}_modified.png'.format(i*config.batch_size+j, object_idx_to_name[cur_obj_success]))
                    imwrite(img_path_red, img_shift_y)

                    img_rec_np = img_rec_y[j].numpy().transpose(1, 2, 0)
                    img_path = os.path.join(result_save_dir, 'img{:06d}_rec_modified.png'.format(i * config.batch_size + j, , object_idx_to_name[cur_obj_success]))
                    imwrite(img_path, img_rec_np)

                    img_rand_np = img_rand_y[j].numpy().transpose(1, 2, 0)
                    img_path = os.path.join(result_save_dir, 'img{:06d}_rand_modified.png'.format(i * config.batch_size + j,,
                                            object_idx_to_name[cur_obj_success]))
                    imwrite(img_path, img_rand_np)

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
        print((count[:, 3] > 0).sum() / count.shape[0])

if True:
    # __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration
    path = '/home/mark1123/data/mark1123'
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
