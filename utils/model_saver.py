import os
import re
import torch


def load_model(model, model_dir=None, appendix=None, epoch='l'):

    load_epoch = None
    load_model = None

    if epoch == 's' or not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        load_epoch = 0
        if not os.path.isdir(model_dir):
            print('models dir not exist')
        elif len(os.listdir(model_dir)) == 0:
            print('models dir is empty')

        print('train from scratch.')
        return load_epoch

    # load latest epoch
    if epoch == 'l':
        for file in os.listdir(model_dir):
            if appendix is not None and appendix not in file:
                continue

            if file.endswith('.pkl'):
                current_epoch = re.search('epoch-\d+', file).group(0).split('-')[1]

                if len(current_epoch) > 0:
                    current_epoch = int(current_epoch)

                    if load_epoch is None or current_epoch > load_epoch:
                        load_epoch = current_epoch
                        load_model = os.path.join(model_dir, file)
                else:
                    continue

        print('load from epoch: %d' % load_epoch)
        model.load_state_dict(torch.load(load_model))

        return load_epoch
    # from given epoch
    else:
        epoch = int(epoch)
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                current_epoch = re.search('epoch-\d+', file).group(0).split('-')[1]
                if len(current_epoch) > 0:
                    if int(current_epoch) == epoch:
                        load_epoch = epoch
                        load_model = os.path.join(model_dir, file)
                        break
        if load_model:
            model.load_state_dict(torch.load(load_model))
            print('load from epoch: %d' % load_epoch)
        else:
            load_epoch = 0
            print('there is not saved models of epoch %d' % epoch)
            print('train from scratch.')
        return load_epoch


def save_model(model, model_dir=None, appendix=None, epoch=1, save_num=5):
    epoch_idx = range(epoch, epoch - save_num, -1)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            current_epoch = re.search('epoch-\d+', file).group(0).split('-')[1]
            if len(current_epoch) > 0:
                if int(current_epoch) not in epoch_idx:
                    os.remove(os.path.join(model_dir, file))
            else:
                continue

    if appendix:
        model_name = os.path.join(model_dir, 'epoch-%d_%s.pkl' % (epoch, appendix))
    else:
        model_name = os.path.join(model_dir, 'epoch-%d.pkl' % epoch)
    torch.save(model.state_dict(), model_name)
