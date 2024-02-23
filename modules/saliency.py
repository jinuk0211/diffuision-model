# 먼저 Saliency detection이란 관심있는
# 물체를 관심이 없는 배경(background)로 부터 분리시키는 것을 말함
#inpainting
import torch
import torchvision
from torch import nn
from PIL import Image
import numpy as np
import os


# MICRO RESNET
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.resblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        out = self.resblock(x)
        return out + x


class Upsample2d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample2d, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class MicroResNet(nn.Module):
    def __init__(self):
        super(MicroResNet, self).__init__()

        self.downsampler = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 8, kernel_size=9, stride=4),
            nn.InstanceNorm2d(8, affine=True),
            # affine=각 채널별로 스케일(scale)과 편향(bias) 파라미터
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
        )

        self.residual = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False, groups=32),
            ResBlock(64),
        )
        # 이미지에서 사람의 시선을 끌고 주목할 만한 부분을 정확하게 추출하는 역할
        self.segmentator = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            Upsample2d(scale_factor=2),
            nn.ReflectionPad2d(4),
            nn.Conv2d(16, 1, kernel_size=9),
            nn.Sigmoid()
        )
# InstanceNorm2d: 
# 각 픽셀의 특징을 개별적으로 정규화하여 배경과 객체를 더욱 효과적으로 구분합니다.

# Upsample2d:  
# 특징맵의 크기를 2배로 확대하여 이미지 해상도와 일치시키고 세분화 정확도를 높입니다.

# Conv2d with kernel_size=9:
# 넓은 영역의 정보를 고려하여 이미지의 세밀한 부분까지 정확하게 세분화합니다.

# Sigmoid: 
# 마지막 레이어에 Sigmoid 함수를 사용하여 각 픽셀이 객체에 속할 확률을 계산합니다.

    def forward(self, x):
        out = self.downsampler(x)
        out = self.residual(out)
        out = self.segmentator(out)
        return out
