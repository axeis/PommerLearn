"""
Utility methods for building the neural network architectures.
"""
from typing import Any

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish,\
    Module


def get_act(act_type):
    """Wrapper method for different non linear activation functions"""
    if act_type == "relu":
        return ReLU()
    if act_type == "sigmoid":
        return Sigmoid()
    if act_type == "tanh":
        return Tanh()
    if act_type == "lrelu":
        return LeakyReLU(negative_slope=0.2)
    if act_type == "hard_sigmoid":
        return Hardsigmoid()
    if act_type == "hard_swish":
        return Hardswish()
    raise NotImplementedError


class MixConv(Module):
    def __init__(self, in_channels, out_channels, kernels):
        """
        Mix depth-wise convolution layers, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
        :param in_channels: Number of input channels
        :param out_channels: Number of convolutional channels
        :param bn_mom: Batch normalization momentum
        :param kernels: List of kernel sizes to use
        :return: symbol
        """
        super(MixConv, self).__init__()

        self.branches = []
        self.num_splits = len(kernels)

        for kernel in kernels:
            self.branch = Sequential(Conv2d(in_channels=in_channels // self.num_splits,
                                 out_channels=out_channels // self.num_splits, kernel_size=(kernel, kernel),
                                 padding=(kernel//2, kernel//2), bias=False,
                                 groups=out_channels // self.num_splits))
            self.branches.append(self.branch)

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.num_splits == 1:
            return self.branch(x)
        else:
            conv_layers = []
            for xi, branch in zip(torch.split(x, dim=1, split_size_or_sections=self.num_splits), self.branches):
                conv_layers.append(branch(xi))

        return torch.cat(conv_layers, 0)


class _Stem(torch.nn.Module):
    def __init__(self, channels, bn_mom=0.9, act_type="relu", nb_input_channels=34):
        """
        Definition of the stem proposed by the alpha zero authors
        :param channels: Number of channels for 1st conv operation
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param nb_input_channels: Number of input channels of the board representation
        """

        super(_Stem, self).__init__()

        self.body = Sequential(
            Conv2d(in_channels=nb_input_channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1),
                   bias=False),
            BatchNorm2d(momentum=bn_mom, num_features=channels),
            get_act(act_type))

    def forward(self, x):
        """
        Compute forward pass
        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _DepthWiseStem(Module):
    def __init__(self, channels, bn_mom=0.9, act_type="relu", nb_input_channels=34):
        """
        Sames as _Stem() but with group depthwise convolutions
        """
        super(_DepthWiseStem, self).__init__()
        self.body = Sequential(Conv2d(in_channels=nb_input_channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False, groups=channels),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1), padding=(0, 0), bias=True),
                               )

    def forward(self, x):
        """
        Compute forward pass
        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _PolicyHead(Module):
    def __init__(self, board_height=11, board_width=11, channels=256, policy_channels=2, n_labels=4992, bn_mom=0.9, act_type="relu",
                 select_policy_from_plane=False):
        """
        Definition of the value head proposed by the alpha zero authors
        :param policy_channels: Number of channels for 1st conv operation in branch 0
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        """

        super(_PolicyHead, self).__init__()

        self.body = Sequential()
        self.select_policy_from_plane = select_policy_from_plane

        if self.select_policy_from_plane:
            self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels, padding=1, kernel_size=(3, 3), bias=False),
                                   BatchNorm2d(momentum=bn_mom, num_features=channels),
                                   get_act(act_type),
                                   Conv2d(in_channels=channels, out_channels=policy_channels, padding=1, kernel_size=(3, 3), bias=False))
            self.nb_flatten = policy_channels*board_width*policy_channels

        else:
            self.body = Sequential(Conv2d(in_channels=channels, out_channels=policy_channels, kernel_size=(1, 1), bias=False),
                                   BatchNorm2d(momentum=bn_mom, num_features=policy_channels),
                                   get_act(act_type))

            self.nb_flatten = board_height*board_width*policy_channels
            self.body2 = Sequential(Linear(in_features=self.nb_flatten, out_features=n_labels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.select_policy_from_plane:
            return self.body(x).view(-1, self.nb_flatten)
        else:
            x = self.body(x).view(-1, self.nb_flatten)
            return self.body2(x)


class _ValueHead(Module):
    def __init__(self, board_height=11, board_width=11, channels=256, channels_value_head=1, fc0=256, bn_mom=0.9, act_type="relu"):
        """
        Definition of the value head proposed by the alpha zero authors
        :param board_height: Height of the board
        :param board_width: Width of the board
        :param channels: Number of channels as input
        :param channels_value_head: Number of channels for 1st conv operation in branch 0
        :param fc0: Number of units in Dense/Fully-Connected layer
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        """

        super(_ValueHead, self).__init__()

        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels_value_head, kernel_size=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels_value_head),
                               get_act(act_type))

        self.nb_flatten = board_height*board_width*channels_value_head
        self.body2 = Sequential(Linear(in_features=self.nb_flatten, out_features=fc0),
                                get_act(act_type),
                                Linear(in_features=fc0, out_features=1),
                                get_act("tanh"))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        x = self.body(x).view(-1, self.nb_flatten)
        return self.body2(x)