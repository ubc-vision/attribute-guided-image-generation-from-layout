"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
# Add the folder path to the sys.path list
from data.vg_custom_mask import get_dataloader
from models.bilinear import crop_bbox_batch
from utils.model_saver_iter import load_model, save_model
from tensorboardX import SummaryWriter


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure


vg_dir = "~/vg"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
num_classes = None
batch_size = 4
num_epochs = 1000
feature_extract = False
lr = 0.0001

exp_name = "resinet50_vg"
log_save_dir = "~/checkpoints/resinet50_vg/logs/"
model_save_dir = '~/checkpoints/' + exp_name


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, input_size=224, start_epoch=0):
    since = time.time()

    val_acc_history = []

    writer = {"train": SummaryWriter(log_save_dir + exp_name + "_train/"),
              "val": SummaryWriter(log_save_dir + exp_name + "_val/")}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_count = 0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                imgs, objs, boxes, masks, obj_to_img, attribute, masks_shift, boxes_shift = batch
                inputs = crop_bbox_batch(imgs, boxes, obj_to_img, input_size)
                inputs = inputs.to(device)
                labels = objs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_count += labels.size(0)

                if (i+1) % 20 == 0:
                    print('epoch: {:04d} iter: {:08d} loss: {:.4f} accu: {:.4f}'.format(epoch+1, i+1, loss.item(), torch.mean((preds == labels.data).float())))

                # Logging.
                loss_logging = {}
                loss_logging['avg loss'] = loss.item()
                loss_logging['accu'] = torch.mean((preds == labels.data).float())

                if (i + 1) % 100 == 0:
                    for tag, roi_value in loss_logging.items():
                        writer[phase].add_scalar(tag, roi_value, epoch * iters_per_epoch[phase] + i + 1)

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.double() / running_count

            print('================================================================')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val': # and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, model_save_dir, '128_resinet50_vg_best', epoch + 1, save_step=1)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    writer['train'].close()
    writer['test'].close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloaders, criterion, input_size=224):
    since = time.time()

    val_acc_history = []
    phase = 'val'
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_count = 0

    # Iterate over data.
    for i, batch in enumerate(dataloaders[phase]):
        imgs, objs, boxes, masks, obj_to_img, attributes = batch
        inputs = crop_bbox_batch(imgs, boxes, obj_to_img, input_size)
        inputs = inputs.to(device)
        labels = objs.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)


        # statistics
        running_loss += loss.item() * labels.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_count += labels.size(0)

        if (i + 1) % 20 == 0:
            print('loss: {:.4f} accu: {:.4f}'.format(loss.item(), torch.mean((preds == labels.data).float())))

    epoch_loss = running_loss / running_count
    epoch_acc = running_corrects.double() / running_count

    print('================================================================')
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


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


data_loader_train, data_loader_val = get_dataloader(batch_size=batch_size, attribute_embedding=106)

iters_per_epoch = {"train": len(data_loader_train)//batch_size, "val": len(data_loader_val)//batch_size}

# Create training and validation dataloaders
dataloaders_dict = {'train': data_loader_train, 'val': data_loader_val}

num_classes = data_loader_train.dataset.num_objects

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Send the model to GPU
model_ft = model_ft.to(device)

#start_epoch = load_model(model_ft, '/home/zhaobo/HDD/Projects/sig-gcn/layout2im/checkpoints/models/resinet50_vg', 'resinet50_vg_best', iter='l')

start_epoch = load_model(model_ft, model_save_dir, '128_resinet50_vg_best', iter='s')
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name == "inception"), start_epoch=start_epoch)

test_model(model_ft, dataloaders_dict, criterion, input_size=224)
