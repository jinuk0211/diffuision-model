"""
작성자: Zhuo Su, Wenzhe Liu
날짜: Feb 18, 2021"""

import math



import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

nets = {
    'baseline': {
        'layer0': 'cv',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'c-v15': {
        'layer0': 'cd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'a-v15': {
        'layer0': 'ad',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'r-v15': {
        'layer0': 'rd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cv',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cv',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'cvvv4': {
        'layer0': 'cd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'avvv4': {
        'layer0': 'ad',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'ad',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'ad',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'rvvv4': {
        'layer0': 'rd',
        'layer1': 'cv',
        'layer2': 'cv',
        'layer3': 'cv',
        'layer4': 'rd',
        'layer5': 'cv',
        'layer6': 'cv',
        'layer7': 'cv',
        'layer8': 'rd',
        'layer9': 'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
    },
    'cccv4': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'cd',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'cd',
        'layer6': 'cd',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'cd',
        'layer10': 'cd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cv',
    },
    'aaav4': {
        'layer0': 'ad',
        'layer1': 'ad',
        'layer2': 'ad',
        'layer3': 'cv',
        'layer4': 'ad',
        'layer5': 'ad',
        'layer6': 'ad',
        'layer7': 'cv',
        'layer8': 'ad',
        'layer9': 'ad',
        'layer10': 'ad',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'cv',
    },
    'rrrv4': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'cv',
        'layer4': 'rd',
        'layer5': 'rd',
        'layer6': 'rd',
        'layer7': 'cv',
        'layer8': 'rd',
        'layer9': 'rd',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'cv',
    },
    'c16': {
        'layer0': 'cd',
        'layer1': 'cd',
        'layer2': 'cd',
        'layer3': 'cd',
        'layer4': 'cd',
        'layer5': 'cd',
        'layer6': 'cd',
        'layer7': 'cd',
        'layer8': 'cd',
        'layer9': 'cd',
        'layer10': 'cd',
        'layer11': 'cd',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cd',
    },
    'a16': {
        'layer0': 'ad',
        'layer1': 'ad',
        'layer2': 'ad',
        'layer3': 'ad',
        'layer4': 'ad',
        'layer5': 'ad',
        'layer6': 'ad',
        'layer7': 'ad',
        'layer8': 'ad',
        'layer9': 'ad',
        'layer10': 'ad',
        'layer11': 'ad',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'ad',
    },
    'r16': {
        'layer0': 'rd',
        'layer1': 'rd',
        'layer2': 'rd',
        'layer3': 'rd',
        'layer4': 'rd',
        'layer5': 'rd',
        'layer6': 'rd',
        'layer7': 'rd',
        'layer8': 'rd',
        'layer9': 'rd',
        'layer10': 'rd',
        'layer11': 'rd',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'rd',
    },
    'carv4': {
        'layer0': 'cd',
        'layer1': 'ad',
        'layer2': 'rd',
        'layer3': 'cv',
        'layer4': 'cd',
        'layer5': 'ad',
        'layer6': 'rd',
        'layer7': 'cv',
        'layer8': 'cd',
        'layer9': 'ad',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'ad',
        'layer14': 'rd',
        'layer15': 'cv',
    },
}

# cd_conv (channel difference convolution)
# 채널 차이 합성곱 연산입니다.
# 3x3 커널을 사용해야 하며, 팽창은 1 또는 2여야 합니다.
# 패딩은 팽창과 동일해야 합니다.
# 각 채널에 대해 평균 계산된 채널 값을 뺀 후 일반적인 합성곱을 수행합니다.

# 채널 간 정보 상호작용을 강조하여 중요한 정보 손실을 최소화합니다.
# 채널 간 상관관계를 효율적으로 활용하여 특징 추출 성능을 향상


# 2. ad_conv (anti-diagonal convolution)
# 반대 대각선 합성곱 연산입니다.
# 3x3 커널을 사용해야 하며, 팽창은 1 또는 2여야 합니다.
# 패딩은 팽창과 동일해야 합니다.
# 가중치를 시계 방향으로 90도 회전시킨 후 일반적인 합성곱을 수행합니다.

# 기존 합성곱은 수평, 수직 방향 정보만 추출하는 반면,
# 항대각 합성곱은 대각선 방향 정보까지 추출하여 더욱 풍부한 특징

# 3. rd_conv (radial difference convolution)
# 반경 차이 합성곱 연산입니다.
# 3x3 커널을 사용해야 하며, 팽창은 1 또는 2여야 합니다.
# 패딩은 팽창의 두 배여야 합니다.
# 가중치를 특정 방식으로 재배열하여 중심 픽셀을 제외하고 가중치 차이를 사용하여 합성곱을 수행합니다.

# 기존 합성곱은 이미지의 픽셀 값만 고려하는 반면,
#  RDC는 픽셀 간의 거리 정보까지 고려하여 이미지의 원형 구조를 효과적으로 활용

def createConvFunc(op_type):    #op = operater type
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd': # channelwise difference
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:  # memory cache와 유사한 역할
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('impossible to be here unless you force that')
        return None


class Conv2d(nn.Module):   #depthwise seperable conv # 한 개의 필터가 한 개의 채널
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        # 입력 채널(32)과 출력 채널(64)은 모두 4개의 그룹으로 균등하게 나뉘며
        #  (그룹당 8개 채널), 각 그룹에서 독립적으로 합성곱이 수행됩니다.
        # 이는 4개의 8x8x3 필터 세트를 형성
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # weight = uniform(-a * sqrt(k), a * sqrt(k))
        if self.bias is not None:
            fan_in, _ =  nn.init._calculate_fan_in_and_fan_out(self.weight)

# # 예시 컨볼루션 층 정의
# conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# # fan_in, fan_out 계산
# fan_in, fan_out = init._calculate_fan_in_and_fan_out(conv.weight)

# fanin - 27(kernel*채널수) fanout-16

            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) #-b~b사이의 값으로 uniform distribute

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
# CSA (Compact Spatial Attention)**는
# 이미지에서 중요한 공간 정보를 강조하고 불필요한 정보를 억제하는
# 딥러닝 기반의 어텐션 메커니즘
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)
        #bias 0으로 초기화
    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y
# x는 입력 특징 맵이고, y는 시그모이드 함수를 통해 계산된 어텐션 맵.
# 어텐션 맵은 각 픽셀의 중요도를 나타내는 값으로 구성.
# 곱셈 연산은 각 픽셀의 중요도에 따라 입력 특징 맵을 조정.
# 중요도가 높은 픽셀은 강조되고, 중요도가 낮은 픽셀은 억제


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
# 일반적인 합성곱 연산은 필터 크기만큼 입력 이미지를 확장하여 계산을 수행합니다. 이 방식은 정보량을 늘리고 모델 성능을 향상시키지만, 동시에 계산량과 메모리 사용량도 증가시키는 단점이 있습니다.

# CDC는 이러한 문제를 해결하기 위해 다음과 같은 방식을 사용합니다.

# 확장: 필터 크기 대신 확장율을 사용하여 입력 이미지를 확장합니다. 확장율은 필터 크기보다 훨씬 작은 값으로 설정할 수 있습니다.
# 채널 분리: 입력 채널을 그룹으로 나누고 각 그룹에 대해 별도의 필터를 적용합니다. 이 방식은 계산량을 줄이고 메모리 사용량을 효율적으로 관리할 수 있도록 합니다.
# 합성곱: 확장된 입력 이미지와 분리된 필터를 사용하여 합성곱 연산을 수행합니다.

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4

# 각 팽창 합성곱은 다양한 팽창율 (dilation rate)을 사용하여 다양한 크기의 시야 정보를 추출합니다.
# 마지막 단계에서 이러한 정보들을 더하여 이미지의 전체적인 정보를 풍부하게 표현


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)
#feature 채널수를 1로, kernelsize-1 나머지는 변화 x

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PDCBlock_converted(nn.Module):
    """  #PDCblock을 conv로
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """
# CPDC: Channel-wise Parallel Dilation Convolution
# APDC: Angular Parallel Dilation Convolution
# RPDC: Radial Parallel Dilation Convolution

# CPDC: 채널별 병렬 확장 합성곱은 입력 이미지의 각 채널에 대해 별도의 확장율을 적용하여 합성곱 연산을 수행합니다. 이는 다양한 채널에서 다양한 수준의 공간 정보를 추출하는 데 유용합니다.

# APDC: 각도별 병렬 확장 합성곱은 입력 이미지를 각도적으로 회전시킨 후 확장된 이미지에 필터를 적용합니다. 이는 이미지의 방향 정보를 효과적으로 추출하는 데 유용합니다.

# RPDC: 반경별 병렬 확장 합성곱은 입력 이미지를 중심으로 원형으로 확장시킨 후 확장된 이미지에 필터를 적용합니다. 이는 이미지의 중심을 기준으로 하는 공간 정보를 효과적으로 추출하는 데 유용합니다.


    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        # 기존 self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane,
                                        kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        # print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        # if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    # print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs


def pidinet():
    pdcs = config_model('carv4')
    dil = 24  # if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=True)

