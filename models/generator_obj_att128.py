import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bilinear import crop_bbox_batch
from models.initialization import weights_init
from math import *
from models.spade.networks.normalization import SPADE


def get_z_random(batch_size, z_dim, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, z_dim) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, z_dim)
    return z


def transform_z_flat(batch_size, time_step, z_flat, obj_to_img):
    # restore z to batch with padding
    z = torch.zeros(batch_size, time_step, z_flat.size(1)).to(z_flat.device)
    for i in range(batch_size):
        idx = (obj_to_img.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        z[i, :n] = z_flat[idx]
    return z


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                    out_channels=4 * self.hidden_dim,
                                                    kernel_size=self.kernel_size,
                                                    padding=self.padding,
                                                    bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, batch_first=False, bias=True,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class LayoutConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, return_all_layers=False):
        super(LayoutConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size, input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, obj_tensor, obj_to_img, hidden_state=None):
        """

        Parameters
        ----------
        obj_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # split input_tensor into list according to obj_to_img
        O = obj_tensor.size(0)
        previous_img_id = 0

        layouts_list = []
        temp = []
        for i in range(O):
            current_img_id = obj_to_img[i]
            if current_img_id == previous_img_id:
                temp.append(obj_tensor[i])
            else:
                temp = torch.stack(temp, dim=0)
                temp = torch.unsqueeze(temp, 0)
                layouts_list.append(temp)
                temp = [obj_tensor[i]]
                previous_img_id = current_img_id
        # append last one
        temp = torch.stack(temp, dim=0)
        temp = torch.unsqueeze(temp, 0)
        layouts_list.append(temp)

        N = len(layouts_list)
        all_layer_output_list, all_last_state_list = [], []
        for i in range(N):
            obj_tensor = layouts_list[i]
            hidden_state = self._init_hidden(batch_size=obj_tensor.size(0), device=obj_tensor.device)

            layer_output_list = []
            last_state_list = []

            seq_len = obj_tensor.size(1)
            cur_layer_input = obj_tensor

            for layer_idx in range(self.num_layers):

                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(seq_len):
                    h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

            all_layer_output_list.append(layer_output_list)
            all_last_state_list.append(last_state_list)

        # concate last output to form a tensor
        batch_output = []
        for i in range(N):
            batch_output.append(all_last_state_list[i][0][0])
        batch_output = torch.cat(batch_output, dim=0)

        return batch_output

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class CropEncoder(nn.Module):
    def __init__(self, conv_dim=64, z_dim=8, class_num=10):
        # default: (3, 32, 32) -> (256, 8, 8)
        super(CropEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        # (3, 32, 32) -> (64, 32, 32)
        self.c1 = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (64, 32, 32) -> (128, 16, 16)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (128, 16, 16) -> (256, 8, 8)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (256, 8, 8) -> (512, 4, 4)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)
        # (512, 4, 4) -> (1024, 2, 2)
        self.conv5 = nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(conv_dim * 16) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 16, class_num)
        # pool
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 256 -> 8
        self.fc_mu = nn.Linear(conv_dim * 16, z_dim)
        self.fc_logvar = nn.Linear(conv_dim * 16, z_dim)

    def forward(self, imgs, objs=None):
        x = imgs
        x = self.c1(x)
        x = self.bn1(x) if objs is None else self.bn1(x, objs)
        x = self.activation(x)
        x = self.c2(x)
        x = self.bn2(x) if objs is None else self.bn2(x, objs)
        x = self.activation(x)
        x = self.c3(x)
        x = self.bn3(x) if objs is None else self.bn3(x, objs)
        x = self.activation(x)
        x = self.c4(x)
        x = self.bn4(x) if objs is None else self.bn4(x, objs)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.bn5(x) if objs is None else self.bn5(x, objs)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = logvar.mul(0.5).exp_()
        eps = get_z_random(std.size(0), std.size(1)).to(imgs.device)
        z = eps.mul(std).add_(mu)

        return z, mu, logvar


class GlobalEncoder(nn.Module):
    def __init__(self):
        # default: (3, 32, 32) -> (256, 8, 8)
        super(GlobalEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        # (64, 8, 8) -> (128, 4, 4)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        # (128, 4, 4) -> (256, 2, 2)
        self.c2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, h):
        h = self.c1(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.c2(h)

        # (256, 2, 2) -> (256,)
        h = torch.sum(h, dim=(2, 3))

        return h


class LayoutEncoder(nn.Module):
    def __init__(self, conv_dim=64, z_dim=8, obj_att_dim=64, class_num=10, resi_num=6, clstm_layers=3, att_dim=64):
        super(LayoutEncoder, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        # self.embedding = nn.Embedding(class_num, embedding_dim)
        # =======
        #         self.activation = nn.ReLU(inplace=True), device_ids=[3,1, 2])
        #         self.embedding = nn.Embedding(class_num, embedding_dim), device_ids=[2,3,1])
        # >>>>>>> f338fa9... remove cuda 0
        if clstm_layers == 1:
            self.clstm = LayoutConvLSTM(8, 512, [64], (5, 5))
        elif clstm_layers == 2:
            self.clstm = LayoutConvLSTM(8, 512, [64, 64], (5, 5))
        elif clstm_layers == 3:
            self.clstm = LayoutConvLSTM(8, 512, [128, 64, 64], (5, 5))

        layers = []
        # Bottleneck layers.
        for i in range(resi_num):
            layers.append(ResidualBlock(dim_in=64, dim_out=64))

        self.residual = nn.Sequential(*layers)

        # (emb+z+att, 128, 128) -> (64, 128, 128)
        self.c0 = nn.Conv2d(obj_att_dim + z_dim, conv_dim, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (64, 128, 128) -> (128, 64, 64)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (128, 64, 64) -> (256, 32, 32)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (256, 32, 32) -> (512, 16, 16)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)
        # (512, 16, 16) -> (512, 8, 8)
        self.pool = nn.AdaptiveAvgPool2d(8)


    def forward(self, objs_att, masks, obj_to_img, z, objs):
        embeddings_z_att = torch.cat((objs_att, z), dim=1)
        h = embeddings_z_att.view(embeddings_z_att.size(0), embeddings_z_att.size(1), 1, 1) * masks

        h = self.c0(h)
        h = self.bn1(h, objs)
        h = self.activation(h)
        h = self.c2(h)
        h = self.bn2(h, objs)
        h = self.activation(h)
        h = self.c3(h)
        h = self.bn3(h, objs)
        h = self.activation(h)
        h = self.c4(h)
        h = self.bn4(h, objs)

        h = self.pool(h)  # replaced by c5
        # clstm fusion (O, 512, 8, 8) -> (n, 64, 8, 8)
        h = self.clstm(h, obj_to_img)
        # residual block
        h = self.residual(h)

        return h


class Decoder(nn.Module):
    def __init__(self, nf=64, conv_dim=64):
        super(Decoder, self).__init__()

        self.sw, self.sh = 8, 8
        self.h_dim = 64
        self.use_vae = False
        z_dim = 8

        self.activation = nn.ReLU(inplace=True)
        #     # (64+128, 8, 8) -> (256, 8, 8)
        self.c0_new = nn.Conv2d(conv_dim + 128, conv_dim * 4, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.spade_0 = SPADE(conv_dim * 4, self.h_dim)
        #     # (256, 8, 8) -> (256, 16, 16)
        self.dc1 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(conv_dim * 4)
        self.spade_1 = SPADE(conv_dim * 4, self.h_dim)
        #     # (256, 16, 16) -> (128, 32, 32)
        self.dc2 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        #     self.bn2 = nn.BatchNorm2d(conv_dim * 2)
        self.spade_2 = SPADE(conv_dim * 2, self.h_dim)
        #     # (128, 32, 32) -> (64, 64, 64)
        self.dc3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, kernel_size=4, stride=2, padding=1, bias=False)
        #     self.bn3 = nn.BatchNorm2d(conv_dim * 1)
        self.spade_3 = SPADE(conv_dim * 1, self.h_dim)
        #     # (64, 64, 64) -> (3, 64, 64)
        self.c4 = nn.Conv2d(conv_dim * 1, 3, kernel_size=7, stride=1, padding=3, bias=True)
        #
        # 2x upsampling
        #     # (128, 64, 64) -> (64, 128, 128)
        #     # self.dc4 = nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, kernel_size=4, stride=2, padding=1, bias=False)
        #     # self.bn4 = nn.BatchNorm2d(conv_dim * 1)
        #     # self.conv = ConvBlock(in_channel=3, out_channel=64, kernel_size=3, padding=1)
        #     # (3, 128, 128) -> (128, 128, 128)
        self.c5 = nn.Conv2d(3, conv_dim * 2, kernel_size=7, stride=1, padding=3, bias=False)
        #     self.bn4 = nn.BatchNorm2d(conv_dim * 2)
        self.spade_4 = SPADE(conv_dim * 2, self.h_dim)
        #     # (128, 128, 128) -> (128, 128, 128)
        self.c6 = nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=5, stride=1, padding=2, bias=False)
        #     self.bn5 = nn.BatchNorm2d(conv_dim * 2)
        self.spade_5 = SPADE(conv_dim * 2, self.h_dim)
        #     # (128, 128, 128) -> (3, 128, 128)
        self.c7 = nn.Conv2d(conv_dim * 2, 3, kernel_size=7, stride=1, padding=3, bias=True)

    #
    def forward(self, hidden, global_h, z=None):
        seg = hidden
 
        global_h = global_h.unsqueeze(-1)
        global_h = global_h.unsqueeze(-1)
        global_h = global_h.expand(-1, -1, 8, 8)
        embeddings_h = torch.cat((hidden, global_h), dim=1)
        h = embeddings_h
        h = self.c0_new(h)
        #     h = self.bn0(h)
        h = self.spade_0(h, seg)
        h = self.activation(h)
        h = self.dc1(h)
        #     h = self.bn1(h)
        h = self.spade_1(h, seg)
        h = self.activation(h)
        h = self.dc2(h)
        #     h = self.bn2(h)
        h = self.spade_2(h, seg)
        h = self.activation(h)
        h = self.dc3(h)
        #     h = self.bn3(h)
        h = self.spade_3(h, seg)
        h = self.activation(h)
        h_64 = self.c4(h)
        #     # h = self.dc4(h)

        # 2x upsampling
        upsample = F.interpolate(h_64, scale_factor=2, mode='nearest')
        h = self.c5(upsample)
        #     h = self.bn4(h)
        h = self.spade_4(h, seg)
        h = self.activation(h)
        h = self.c6(h)
        #     h = self.bn5(h)
        h = self.spade_5(h, seg)
        h = self.activation(h)
        h = self.c7(h)
        #
        #     # fade in
        #     # alpha = 0.7     # used to be 0.7
        #     # skip_rgb = self.to_rgb0(upsample)
        #     # h = (1 - alpha) * skip_rgb + alpha * h
        #
        return h


class AttributeEncoder(nn.Module):
    def __init__(self, attribute_dim=106, embedding_dim = 64, class_num = 10):
        super(AttributeEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        # (1024, 8, 8) -> (256, 8, 8)
        self.embedding = nn.Embedding(class_num, embedding_dim)

        self.c0 = nn.Linear(attribute_dim + embedding_dim, 128)
        self.bn0 = nn.BatchNorm1d(128)
        self.c1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.c2 = nn.Linear(64, 64)

    def forward(self, objs, attribute):
        embeddings = self.embedding(objs)
        embeddings_att = torch.cat((embeddings, attribute), dim=1)
        a = embeddings_att
        a = self.c0(a)
        a = self.bn0(a)
        a = self.activation(a)
        a = self.c1(a)
        a = self.bn1(a)
        a = self.activation(a)
        a = self.c2(a)

        return a


class Generator(nn.Module):
    def __init__(self, num_embeddings, obj_att_dim=64, z_dim=8, obj_size=64, clstm_layers=3, attribute_dim=128):
        super(Generator, self).__init__()
        self.obj_size = obj_size
        # (3, 32, 32) -> (256, 4, 4) -> 8

        self.crop_encoder = CropEncoder(z_dim=z_dim, class_num=num_embeddings)
        self.layout_encoder = LayoutEncoder(z_dim=z_dim, obj_att_dim=obj_att_dim, class_num=num_embeddings,
                                            clstm_layers=clstm_layers)
        self.decoder = Decoder()

        self.global_encoder = GlobalEncoder()
        # self.apply(weights_init)
        self.attribute_encoder = AttributeEncoder(attribute_dim=attribute_dim, embedding_dim=obj_att_dim, class_num=num_embeddings)

    def forward(self, imgs, objs, boxes, masks, obj_to_img, z_rand, attribute, masks_shift, boxes_shift, attribute_est):
        crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, self.obj_size)
        z_rec, mu, logvar = self.crop_encoder(crops_input, objs)

        objs_att = self.attribute_encoder(objs, attribute)
        objs_att_est = self.attribute_encoder(objs, attribute_est)
        # (n, clstm_dim*2, 8, 8)

        h_rec = self.layout_encoder(objs_att_est, masks, obj_to_img, z_rec, objs)
        h_rand = self.layout_encoder(objs_att, masks, obj_to_img, z_rand, objs)
        h_shift = self.layout_encoder(objs_att, masks_shift, obj_to_img, z_rand, objs)

        # global context encoder
        h_rec_global = self.global_encoder(h_rec)
        h_rand_global = self.global_encoder(h_rand)
        h_shift_global = self.global_encoder(h_shift)

        img_rec = self.decoder(h_rec, h_rec_global)
        img_rand = self.decoder(h_rand, h_rand_global)
        img_shift = self.decoder(h_shift, h_shift_global)

        crops_rand = crop_bbox_batch(img_rand, boxes, obj_to_img, self.obj_size)
        _, z_rand_rec, _ = self.crop_encoder(crops_rand, objs)

        crops_input_rec = crop_bbox_batch(img_rec, boxes, obj_to_img, self.obj_size)

        crops_shift = crop_bbox_batch(img_shift, boxes_shift, obj_to_img, self.obj_size)
        _, z_rand_shift, _ = self.crop_encoder(crops_shift, objs)

        return crops_input, crops_input_rec, crops_rand, crops_shift, img_rec, img_rand, img_shift, mu, logvar, z_rand_rec, z_rand_shift


if __name__ == '__main__':
    from data.vg_custom_mask import get_dataloader

    device = torch.device('cuda:0')
    z_dim = 8
    batch_size = 4

    train_loader, _ = get_dataloader(batch_size=batch_size)
    vocab_num = train_loader.dataset.num_objects

    # test Generator
    model = Generator(num_embeddings=vocab_num, z_dim=z_dim).to(device)

    for i, batch in enumerate(train_loader):
        print(i)
        imgs, objs, boxes, masks, obj_to_img, att = batch
        z = torch.randn(objs.size(0), z_dim).to(device)
        imgs, objs, boxes, masks, obj_to_img, att = imgs.to(device), objs.to(device), boxes.to(device), masks.to(
            device), obj_to_img.to(device), att.to(device)

        outputs = model(imgs, objs, boxes, masks, obj_to_img, z, att)

        for output in outputs:
            print(output.shape)

        if i == 10: break