import torch
from torch import nn
import torch.nn.functional as F
import math
import functools
from einops import rearrange, repeat
import itertools
import torchvision
from model.head_animation.EMOP import args as args_utils

def replace_conv_to_ws_conv(module, conv2d=True, conv3d =True):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    prev_prev_attr = None
    prev_attr = None
    for indx, (attr_str, _) in enumerate(module.named_children()):

        if indx == 0:
            prev_prev_attr = getattr(module, attr_str)
        elif indx == 1:
            prev_attr = getattr(module, attr_str)
        else:
            # print(type(target_attr))
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.Conv2d and conv2d and (type(prev_prev_attr) == torch.nn.GroupNorm or type(prev_attr) == torch.nn.GroupNorm): #
                new_conv = Conv2d_ws(target_attr.in_channels, target_attr.out_channels, kernel_size=target_attr.kernel_size, stride = target_attr.stride, padding = target_attr.padding, dilation = target_attr.dilation,
                                     groups=target_attr.groups, bias=True)
                setattr(module, attr_str, new_conv)

            if type(target_attr) == torch.nn.Conv3d and conv3d and (type(prev_prev_attr) == AdaptiveGroupNorm or type(prev_attr) == AdaptiveGroupNorm): #
                new_conv = Conv3d_ws(target_attr.in_channels, target_attr.out_channels, kernel_size=target_attr.kernel_size, stride = target_attr.stride, padding = target_attr.padding, dilation = target_attr.dilation,
                                     groups=target_attr.groups, bias=True)
                setattr(module, attr_str, new_conv)
            prev_prev_attr = prev_attr
            prev_attr = target_attr

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_conv_to_ws_conv(immediate_child_module, name)

    return module

def apply_ws_to_nets(obj):
    # ws_nets_names = args_utils.parse_str_to_list(obj.args.ws_networks, sep=',')
    ws_networks='local_encoder_nw, local_encoder_seg_nw, local_encoder_mask_nw, idt_embedder_nw, expression_embedder_nw, xy_generator_nw, uv_generator_nw, warp_embed_head_orig_nw,  pose_embed_decode_nw, pose_embed_code_nw, volume_process_nw, volume_source_nw, volume_pred_nw, decoder_nw, backgroung_adding_nw, background_process_nw'
    ws_networks='app_encoder, idt_encoder, expression_encoder, decoder, warp_embed_head_orig_nw, src2ref, ref2tgt, volume_source_nw, volume_process_nw'
    ws_nets_names = args_utils.parse_str_to_list(ws_networks, sep=',')

    for net_name in ws_nets_names:
        try:
            net = getattr(obj, net_name)
            # import pdb; pdb.set_trace()
            new_net = replace_conv_to_ws_conv(net, conv2d=True, conv3d=True)
            setattr(obj, net_name, new_net)
            # print(f'WS applied to {net_name}')
        except Exception as e:
            pass



############################################################
#                Definitions for the layers                #
############################################################
class Conv2d_ws(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_ws, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv3d_ws(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d_ws, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        w = w - w_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1,1,1,1,1) + 1e-5
        w = w / std.expand_as(w)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            conv_layer_type: str = 'conv',
            norm_layer_type: str = 'bn',
            activation_type: str = 'relu',
            resize_layer_type: str = 'none',
            efficient_upsampling: bool = False,  # place upsampling layer before the second convolution
            return_feats: bool = False,  # return feats after the first convolution,
    ):
        """This is a base module for residual blocks"""
        super(ResBlock, self).__init__()
        # Initialize layers in the block
        self.return_feats = return_feats

        m_bias= False
        if resize_layer_type in ['nearest', 'bilinear', 'blur']:
            self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=stride, mode=resize_layer_type)
            self.efficient_upsampling = efficient_upsampling
            if resize_layer_type=='blur':
                self.upsample = Upsample_sg2(kernel=[1, 3, 3, 1])

        downsample = resize_layer_type in downsampling_layers and stride > 1
        if downsample:
            downsampling_layer = downsampling_layers[resize_layer_type]

        normalize = norm_layer_type != 'none'
        if normalize:
            norm_layer = norm_layers[norm_layer_type]

        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]

        if '3d' in conv_layer_type:
            num_kernel_dims = 3
        else:
            num_kernel_dims = 2

        ### Initialize the layers of the first half of the block ###
        layers = []

        if normalize:
            layers += [norm_layer(in_channels)]

        layers += [
            activation(inplace=True),
            conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if normalize:
            layers += [norm_layer(out_channels)]

        layers += [activation(inplace=True)]

        self.block_feats = nn.Sequential(*layers)

        layers = [
            conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if downsample:
            layers += [downsampling_layer(stride)]

        self.block = nn.Sequential(*layers)

        ### Initialize a skip connection block, if needed ###
        if in_channels != out_channels or downsample:
            layers = []

            if in_channels != out_channels:
                layers += [conv_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,) * num_kernel_dims,
                    bias=m_bias)]

            if downsample:
                layers += [downsampling_layer(stride)]

            self.skip = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = inputs

        if hasattr(self, 'upsample') and not self.efficient_upsampling:
            outputs = self.upsample(inputs)

        feats = self.block_feats(outputs)
        outputs = feats

        if hasattr(self, 'upsample') and self.efficient_upsampling:
            outputs = self.upsample(feats)

        outputs_main = self.block(outputs)

        outputs_skip = inputs

        if hasattr(self, 'upsample'):
            outputs_skip = self.upsample(inputs)

        if hasattr(self, 'skip'):
            outputs_skip = self.skip(outputs_skip)

        outputs = outputs_main + outputs_skip

        if self.return_feats:
            return outputs, feats
        else:
            return outputs


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            conv_layer_type: str = 'conv',
            norm_layer_type: str = 'none',
            activation_type: str = 'relu',
            resize_layer_type: str = 'none',
            efficient_upsampling: bool = False,  #
            return_feats: bool = False,
    ):
        """This is a base module for residual blocks"""
        super(ConvBlock, self).__init__()
        # Initialize layers in the block
        self.return_feats = return_feats
        m_bias = False

        if resize_layer_type in ['nearest', 'bilinear'] and stride > 1:
            self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=stride, mode=resize_layer_type)

        downsample = resize_layer_type in downsampling_layers and stride > 1
        if downsample:
            downsampling_layer = downsampling_layers[resize_layer_type]

        normalize = norm_layer_type != 'none'
        if normalize:
            norm_layer = norm_layers[norm_layer_type]

        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]

        if '3d' in conv_layer_type:
            num_kernel_dims = 3
        else:
            num_kernel_dims = 2

        ### Initialize the layers of the first half of the block ###
        layers = [
            conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if normalize:
            layers += [norm_layer(out_channels)]

        layers += [activation(inplace=True)]

        self.block = nn.Sequential(*layers)

        if downsample:
            self.downsample = downsampling_layer(stride)

    def assign_spade_feats(self, feats):
        for m in self.modules():
            if m.__class__.__name__ == 'AdaptiveSPADE':
                m.feats = feats

    def forward(self, inputs, spade_feats=None):
        if spade_feats is not None:
            self.assign_spade_feats(spade_feats)

        if hasattr(self, 'upsample'):
            outputs = self.upsample(inputs)
        else:
            outputs = inputs

        feats = self.block(outputs)

        if hasattr(self, 'downsample'):
            outputs = self.downsample(feats)
        else:
            outputs = feats

        if self.return_feats:
            return outputs, feats
        else:
            return outputs


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_height, in_width = inputs.size()

        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, out_height, self.upscale_factor,
            out_width, self.upscale_factor)

        channels *= self.upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(AdaptiveConv, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert not bias, 'bias == True is not supported for AdaptiveConv'
        self.bias = None

        self.kernel_numel = kernel_size[0] * kernel_size[1]
        if len(kernel_size) == 3:
            self.kernel_numel *= kernel_size[2]

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.ada_weight = None  # assigned externally

        if len(kernel_size) == 2:
            self.conv_func = F.conv2d
        elif len(kernel_size) == 3:
            self.conv_func = F.conv3d

    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.ada_weight.type():
            weight = self.ada_weight.type(inputs.type())
        else:
            weight = self.ada_weight

        # Conv is applied to the inputs grouped by t frames
        B = weight.shape[0]
        T = inputs.shape[0] // B
        assert inputs.shape[0] == B * T, 'Wrong shape of weight'

        if self.kernel_numel > 1:
            if weight.shape[0] == 1:
                # No need to iterate through batch, can apply conv to the whole batch
                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)

            else:
                outputs = []
                for b in range(B):
                    outputs += [self.conv_func(inputs[b * T:(b + 1) * T], weight[b], None, self.stride, self.padding,
                                               self.dilation, self.groups)]
                outputs = torch.cat(outputs, 0)

        else:
            if weight.shape[0] == 1:
                if len(inputs.shape) == 5:
                    weight = weight[..., None, None, None]
                else:
                    weight = weight[..., None, None]

                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)
            else:
                # 1x1(x1) adaptive convolution is a simple bmm
                if len(weight.shape) == 6:
                    weight = weight[..., 0, 0, 0]
                else:
                    weight = weight[..., 0, 0]

                outputs = torch.bmm(weight, inputs.view(B * T, inputs.shape[1], -1)).view(B, -1, *inputs.shape[2:])

        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)



def replace_bn_to_in(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            new_bn = torch.nn.InstanceNorm2d(target_attr.num_features, target_attr.eps,
                                             target_attr.momentum, target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_in(immediate_child_module, name)

    return module


def replace_bn_to_gn(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d or type(target_attr) == torch.nn.InstanceNorm2d:
            new_bn = torch.nn.GroupNorm(32, target_attr.num_features, target_attr.eps, target_attr.affine)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_gn(immediate_child_module, name) #TODO поменять на GN

    return module

def replace_bn_to_bcn(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d or type(target_attr) == torch.nn.InstanceNorm2d:
            new_bn = BCNorm(32, target_attr.num_features, target_attr.eps)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_bcn(immediate_child_module, name) #TODO поменять на GN

    return module

class ProjectorNorm(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_embed_size,
                 gen_max_channels):
        super(ProjectorNorm, self).__init__()
        self.eps = eps

        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm'] :
                self.u += [nn.Parameter(torch.empty(m.num_features, gen_max_channels))]
                self.v += [nn.Parameter(torch.empty(gen_embed_size ** 2, 2))]

                nn.init.uniform_(self.u[-1], a=-math.sqrt(3 / gen_max_channels),
                                 b=math.sqrt(3 / gen_max_channels))
                nn.init.uniform_(self.v[-1], a=-math.sqrt(3 / gen_embed_size ** 2),
                                 b=math.sqrt(3 / gen_embed_size ** 2))


    def forward(self, embed_dict, iter=0):
        params = []

        for u, v in zip(self.u, self.v):
            # print(u.shape, v.shape)
            embed = embed_dict['orig']

            param = u[None].matmul(embed).matmul(v[None])
            weight, bias = param.split(1, dim=2)

            params += [(weight[..., 0], bias[..., 0])]

        # import pdb; pdb.set_trace()
        return params


class ProjectorConv(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_adaptive_kernel,
                 gen_max_channels):
        super(ProjectorConv, self).__init__()
        self.eps = eps
        self.adaptive_kernel = gen_adaptive_kernel

        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.kernel_size = []

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ == 'AdaptiveConv':
                # Assumes that adaptive conv layers have no bias
                kernel_numel = m.kernel_size[0] * m.kernel_size[1]
                if len(m.kernel_size) == 3:
                    kernel_numel *= m.kernel_size[2]

                if kernel_numel == 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, gen_max_channels // 2))]
                    self.v += [nn.Parameter(torch.empty(gen_max_channels // 2, m.in_channels))]

                elif kernel_numel > 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, gen_max_channels // 2))]
                    self.v += [nn.Parameter(torch.empty(m.in_channels, gen_max_channels // 2))]

                self.kernel_size += [m.kernel_size]

                bound = math.sqrt(3 / (gen_max_channels // 2))
                nn.init.uniform_(self.u[-1], a=-bound, b=bound)
                nn.init.uniform_(self.v[-1], a=-bound, b=bound)

    def forward(self, embed_dict):
        params = []

        for u, v, kernel_size in zip(self.u, self.v, self.kernel_size):
            kernel_numel = kernel_size[0] * kernel_size[1]
            if len(kernel_size) == 3:
                kernel_numel *= kernel_size[2]

            if kernel_numel == 1:
                embed = embed_dict['fc']
            else:
                if self.adaptive_kernel:
                    if kernel_numel == 9:
                        embed = embed_dict['conv2d']
                    elif kernel_numel == 27:
                        embed = embed_dict['conv3d']
                    embed = embed.view(embed.shape[0], embed.shape[1], -1, kernel_numel)
                else:
                    embed = embed_dict['fc'][..., None]

            if kernel_numel == 1:
                # AdaptiveConv with kernel size = 1
                weight = u[None].matmul(embed).matmul(v[None])
                weight = weight.view(*weight.shape, *kernel_size)  # B x C_out x C_in x 1 ...
            else:
                # AdaptiveConv with kernel size > 1
                if self.adaptive_kernel:
                    kernel_numel_ = kernel_numel
                    kernel_size_ = kernel_size
                else:
                    kernel_numel_ = 1
                    kernel_size_ = (1,) * len(kernel_size)

                param = embed.view(*embed.shape[:2], -1)
                param = u[None].matmul(param)  # B x C_out x C_emb/2
                b, c_out = param.shape[:2]
                param = param.view(b, c_out, -1, kernel_numel_)
                param = v[None].matmul(param)  # B x C_out x C_in x kernel_numel
                weight = param.view(*param.shape[:3], *kernel_size_)

            params += [weight]

        return params


def assign_adaptive_conv_params(net_or_nets, params, adaptive_conv_type, alpha_conv=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name == 'AdaptiveConv':
            attr_name = 'weight_orig' if hasattr(m, 'weight_orig') else 'weight'
            weight = getattr(m, attr_name)
            ada_weight = params.pop(0)

            if adaptive_conv_type == 'sum':
                ada_weight = weight[None] + ada_weight * alpha_conv
            elif adaptive_conv_type == 'mul':
                ada_weight = weight[None] * (torch.sigmoid(ada_weight) * alpha_conv + (1 - alpha_conv))

            setattr(m, 'ada_' + attr_name, ada_weight)

def assign_adaptive_norm_params(net_or_nets, params, alpha_norm=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm']:  #TODO разобраться
            ada_weight, ada_bias = params.pop(0)

            m.ada_weight = m.weight[None] + ada_weight * alpha_norm
            m.ada_bias = m.bias[None] + ada_bias * alpha_norm


class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs):
        outputs = super(AdaptiveGroupNorm, self).forward(inputs)
        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]
        outputs = outputs.view(B * T, *outputs.shape[2:])
        # print(inputs.shape, outputs.shape)
        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
               'affine=True'.format(**self.__dict__)


class ProjectorNormLinear(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_embed_size,
                 gen_max_channels,
                 emb_v_exp=False,
                 no_detach_frec=1,
                 key_emb = 'orig'):
        super(ProjectorNormLinear, self).__init__()
        self.eps = eps
        self.emb_v_exp = emb_v_exp
        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.no_detach_frec = no_detach_frec

        self.key_emb = key_emb

        input_n = 512 if emb_v_exp else 512*16
        self.fc = nn.Sequential(
                    nn.Linear(input_n, 512, bias=False),
                    nn.ReLU(),
                    nn.Linear(512, 512*2, bias=False))

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm'] :
                self.u += [nn.Parameter(torch.empty(m.num_features, 512))]
                self.v += [nn.Parameter(torch.empty(2, 2))]

                nn.init.uniform_(self.u[-1], a=-math.sqrt(3 / 512),
                                 b=math.sqrt(3 / 512))
                nn.init.uniform_(self.v[-1], a=-math.sqrt(3 / 2 ),
                                 b=math.sqrt(3 / 2))

    def forward(self, embed_dict, iter=0):
        params = []
        if self.emb_v_exp:
            embed = embed_dict['ada_v'].detach()
        else:
            embed = embed_dict[self.key_emb].view(-1, 512*16) if iter%self.no_detach_frec==0 else embed_dict[self.key_emb].view(-1, 512*16).detach()


        embed = self.fc(embed).view(-1, 512, 2)

        for u, v in zip(self.u, self.v):

            param = u[None].matmul(embed).matmul(v[None])
            weight, bias = param.split(1, dim=2)

            params += [(weight[..., 0], bias[..., 0])]

        return params


class Upsample_sg2(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class GridSample(nn.Module):
    def __init__(self, size):
        super(GridSample, self).__init__()
        self.size = size
        self.register_backward_hook(scale_warp_grad_norm)

    def forward(self, inputs, grid, padding_mode='reflection'):
        return F.grid_sample(inputs, grid, padding_mode=padding_mode)


# Supported blocks
blocks = {
    'res': ResBlock,
    'conv': ConvBlock
}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'avgpool_3d': nn.AvgPool3d,
    'maxpool_3d': nn.MaxPool3d,
    'pixelunshuffle': PixelUnShuffle}

# Supported normalization layers
norm_layers = {
    'in': lambda num_features, affine=True: nn.InstanceNorm2d(num_features=num_features, affine=affine),
    'bn': lambda num_features: nn.BatchNorm2d(num_features=num_features, momentum=MOMENTUM),
    'bn_3d': lambda num_features: nn.BatchNorm3d(num_features=num_features, momentum=MOMENTUM),
    'in_3d': lambda num_features, affine=True: nn.InstanceNorm3d(num_features=num_features, affine=affine),
    'sync_bn': lambda num_features: nn.SyncBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'ada_in': lambda num_features, affine=True: AdaptiveInstanceNorm(num_features=num_features, affine=affine),
    'ada_bn': lambda num_features: AdaptiveBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'ada_sync_bn': lambda num_features: AdaptiveSyncBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=32, num_channels=num_features, affine=affine),
    'bcn': lambda num_features, affine=True: BCNorm(num_channels=num_features, num_groups=32, estimate=True),
    'bcn_3d': lambda num_features, affine=True: BCNorm(num_channels=num_features, num_groups=32,  estimate=True),
    'gn_24': lambda num_features, affine=True: nn.GroupNorm(num_groups=24, num_channels=num_features, affine=affine),
    'gn_3d': lambda num_features, affine=True: nn.GroupNorm(num_groups=32, num_channels=num_features, affine=affine),
    'ada_gn': lambda num_features, affine=True: AdaptiveGroupNorm(num_groups=32, num_features=num_features, affine=affine),
    # 'ada_gn': lambda num_features, affine=True: AdaptiveInstanceNorm(num_features=num_features, affine=affine),
    # 'ada_bcn': lambda num_features, affine=True: AdaptiveGroupNorm(num_groups=32, num_features=num_features, affine=affine),
    'ada_bcn': lambda num_features, affine=True: AdaptiveBCNorm(num_groups=32, num_features=num_features, estimate=True)
}

# Supported activations
activations = {
    'relu': nn.ReLU,
    # 'relu': functools.partial(nn.LeakyReLU, negative_slope=0.04),
    'lrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}

# Supported conv layers
conv_layers = {
    'conv': nn.Conv2d,
    # 'conv': Conv2d_ws,
    'conv_3d': nn.Conv3d,
    # 'conv_3d': Conv3d_ws,
    'ada_conv': AdaptiveConv,
    'ada_conv_3d': AdaptiveConv}
