"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from data.vg_custom_mask import get_dataloader
from models.bilinear import crop_bbox_batch
from utils.model_saver_iter import load_model, save_model
import pickle


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure

data_dir = '~/pickle/128_vg_pkl_resinet50/'
vg_dir = "~/vg"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
num_classes = None
batch_size = 32
num_epochs = 1000
feature_extract = False

exp_name = "resinet50_model_vg"
# log_save_dir = "'/home/mark1123/projects/rrg-lsigal/mark1123/layout2im/"
model_save_dir = '~/checkpoints/resinet50_vg/crop=224'


def test_model(model, pickle_files, criterion, input_size=224):
    since = time.time()

    val_acc_history = []
    phase = 'val'
    model.eval()

    running_loss_real = 0.0
    running_corrects_real = 0
    running_count_real = 0

    running_loss_fake = 0.0
    running_corrects_fake = 0
    running_count_fake = 0

    running_loss_fake_shift = 0.0
    running_corrects_fake_shift = 0
    running_count_fake_shift = 0

    # Iterate over data.
    for i, pickle_file in enumerate(pickle_files):

        batch = pickle.load(open(pickle_file, 'rb'))
        imgs, imgs_rand, imgs_shift, objs, boxes, boxes_shift, obj_to_img = batch['imgs'], batch['imgs_rand'], batch['imgs_shift'], batch['objs'], batch['boxes'], batch['boxes_shift'], batch['obj_to_img']

        inputs_real = crop_bbox_batch(imgs, boxes, obj_to_img, input_size)
        inputs_real = inputs_real.to(device)

        inputs_fake = crop_bbox_batch(imgs_rand, boxes, obj_to_img, input_size)
        inputs_fake = inputs_fake.to(device)

        inputs_fake_shift = crop_bbox_batch(imgs_shift, boxes_shift, obj_to_img, input_size)
        inputs_fake_shift = inputs_fake_shift.to(device)

        labels = objs.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs_real = model(inputs_real)
            loss_real = criterion(outputs_real, labels)
            _, preds_real = torch.max(outputs_real, 1)

            outputs_fake = model(inputs_fake)
            loss_fake = criterion(outputs_fake, labels)
            _, preds_fake = torch.max(outputs_fake, 1)

            outputs_fake_shift = model(inputs_fake_shift)
            loss_fake_shift = criterion(outputs_fake_shift, labels)
            _, preds_fake_shift = torch.max(outputs_fake_shift, 1)

        # statistics
        running_loss_real += loss_real.item() * labels.size(0)
        running_corrects_real += torch.sum(preds_real == labels.data)
        running_count_real += labels.size(0)

        running_loss_fake += loss_fake.item() * labels.size(0)
        running_corrects_fake += torch.sum(preds_fake == labels.data)
        running_count_fake += labels.size(0)

        running_loss_fake_shift += loss_fake_shift.item() * labels.size(0)
        running_corrects_fake_shift += torch.sum(preds_fake_shift == labels.data)
        running_count_fake_shift += labels.size(0)

        if (i + 1) % 20 == 0:
            print('real loss: {:.4f} accu: {:.4f} fake loss: {:.4f} accu: {:.4f} shift loss: {:.4f} accu: {:.4f}'.format(loss_real.item(),
                                                                                         torch.mean((preds_real == labels.data).float()),
                                                                                         loss_fake.item(),
                                                                                         torch.mean((preds_fake == labels.data).float()),
                                                                                         loss_fake_shift.item(),
                                                                                         torch.mean((preds_fake_shift == labels.data).float()))
                  )

    epoch_loss_real = running_loss_real / running_count_real
    epoch_acc_real = running_corrects_real.double() / running_count_real
    epoch_loss_fake = running_loss_fake / running_count_fake
    epoch_acc_fake = running_corrects_fake.double() / running_count_fake
    epoch_loss_fake_shift = running_loss_fake_shift / running_count_fake_shift
    epoch_acc_fake_shift = running_corrects_fake_shift.double() / running_count_fake_shift

    print('================================================================')
    print('{} Real Loss: {:.4f} Acc: {:.4f} Fake Loss: {:.4f} Acc: {:.4f} shift loss: {:.4f} accu: {:.4f}'.format(phase,
                                                                                  epoch_loss_real,
                                                                                  epoch_acc_real,
                                                                                  epoch_loss_fake,
                                                                                  epoch_acc_fake,
                                                                                  epoch_loss_fake_shift,
                                                                                  epoch_acc_fake_shift
                                                                                  ))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

pickle_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
print(pickle_files)
data_loader_train, data_loader_val = get_dataloader(batch_size=batch_size, VG_DIR=vg_dir)
# Create training and validation dataloaders
dataloaders_dict = {'train': data_loader_train, 'val': data_loader_val}

num_classes = data_loader_train.dataset.num_objects

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)
start_epoch = load_model(model_ft, model_save_dir, 'resinet50_vg_best', iter='l')

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Test and evaluate
test_model(model_ft, pickle_files, criterion, input_size=224)

