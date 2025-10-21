import torch.nn as nn
from .attention import AttentionSum 
from .direction_metrics import DirectionAssigned
import numpy as np
import torch
from .encoder2decoder import *
#一个基础block conv+BN+ReLU
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        super(BasicBlock,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forwad(self,x):
        out=self.net(x)
        return out
class BasicLKBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=1) :
        super(BasicLKBlock,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        out=self.net(x)
        return out


class ResBasicBlock(nn.Module):
    def __init__(self,in_channels, kernel_size,stride,padding):
        super(ResBasicBlock,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels,int(in_channels/2),kernel_size,padding,stride),
            nn.BatchNorm2d(int(in_channels/2)),
            nn.LeakyReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(int(in_channels/2),in_channels,kernel_size,padding,stride),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )
    def forward(self,x):
        result=self.layer1(x)
        result=self.layer2(result)
        return result+x

#multidirec+encoder/decoder(LKC)+cbam
class MultiAreaNet(nn.Module):
    def __init__(self,in_channel):
        super(MultiAreaNet,self).__init__()
        #灰度值计算
        self.directionLayerList=[DirectionAssigned(i) for i in range(1,9)]
        self.conv_init=BasicLKBlock(in_channel,8,3,1,1)
        #修改stride2->1
        self.conv_start=BasicLKBlock(8,16,3,1,1)
        self.cbma=AttentionSum(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.encoder_0=self.pileBlock(16,16,block=ResNetBlock)
        self.encoder_1=self.pileBlock(16,32,block=ResNetBlock,blockNumber=2)
        self.encoder_2=self.pileBlock(32,64,block=ResNetBlock,blockNumber=2)
        self.encoder_3=self.pileBlock(64,128,block=ResNetBlock,blockNumber=2)

        self.mid=self.pileBlock(128,256,ResNetBlock,blockNumber=2)

        self.decoder_3=self.pileBlock(256+128,128,ResNetBlock,2)
        self.decoder_2=self.pileBlock(128+64,64,ResNetBlock,2)
        self.decoder_1=self.pileBlock(64+32,32,ResNetBlock,2)
        self.decoder_0=self.pileBlock(32+16,16,ResNetBlock)

        self.output_0=nn.Conv2d(16,1,1)
        self.output_1=nn.Conv2d(32,1,1)
        self.output_2=nn.Conv2d(64,1,1)
        self.output_3=nn.Conv2d(128,1,1)


        self.out_layer=nn.Conv2d(4,1,3,1,1)


        # self.LR1=BasicBlock(1,16,3,1,1)
        # self.LR2=BasicBlock(16,32,3,2,1)
        # #残差结构 四个不同kn_size(2k-1)的feature map连接 k
        # self.res0=self.pileBlock(16,ResBasicBlock,1,1,0)
        # self.res1=self.pileBlock(32,ResBasicBlock,3,1,1,2)
        # self.res2=self.pileBlock(32,ResBasicBlock,5,1,2,2)
        # self.res3=self.pileBlock(32,ResBasicBlock,7,1,3,2)
        #cbam 注意力机制(channel+spatial)
        self.attention=AttentionSum(32,32)
        self.contation=BasicLKBlock(4*32,32,3,1,1)
        self.resSum=BasicLKBlock(16,32,1,1,0)
        

    def forward(self,x,tag):
        #各方向灰度值计算 相反方向进行pointwise mul同时进行求和
        directionResult=[self.directionLayerList[i](x) for i in range(len(self.directionLayerList))]
        directionRes=[directionResult[i].mul(directionResult[i+4]) for i in range(0,4)]
        
        directionUltraRes=directionRes[0]
        for i in range(1,len(directionRes)):
            directionUltraRes+=directionRes[i]
        directionUltraRes=nn.functional.sigmoid(directionUltraRes)
        # nn.functional.sigmoid()
        
        #将heatmap+x进行res操作传入en-de结构
        out1=self.conv_init(x)
        out2=out1.mul(directionUltraRes)
        out=self.conv_start(out1+out2) #得到ht+x结果
        
        en_out0=self.encoder_0(out)  
        en_out1=self.encoder_1(self.pool(en_out0))
        en_out2=self.encoder_2(self.pool(en_out1))
        en_out3=self.encoder_3(self.pool(en_out2))
        mid_out=self.mid(self.pool(en_out3))
        de_out3=self.decoder_3(torch.cat([en_out3,self.up(mid_out)],1))
        de_out2=self.decoder_2(torch.cat([en_out2,self.up(de_out3)],1))
        de_out1=self.decoder_1(torch.cat([en_out1,self.up(de_out2)],1))
        de_out0=self.decoder_0(torch.cat([en_out0,self.up(de_out1)],1))
        #输出
        mask0=self.output_0(de_out0)
        mask1=self.output_1(de_out1)
        mask2=self.output_2(de_out2)
        mask3=self.output_3(de_out3)
        res=torch.cat([mask0,self.up(mask1),self.up_4(mask2),self.up_8(mask3)],dim=1)
        result=self.out_layer(res)

            
        if tag:
            return [mask0,mask1,mask2,mask3],result
        else:
            
            
            return [],result
        
        



    
    #构建一个由多个block堆叠的网络块
    def pileBlock(self,in_channels,out_channels,block,blockNumber=1):
        netLayer=[]
        netLayer.append(block(in_channels,out_channels))
        for i in range(blockNumber-1):
            netLayer.append(block(out_channels,out_channels))
        return nn.Sequential(*netLayer)
        