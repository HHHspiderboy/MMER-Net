from .attention import *
import torch.nn as nn
from .encoder2decoder import *
from .direction_metrics import DirectionAssigned
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
    
class MSHNet(nn.Module):
    def __init__(self, in_channels,block=ResNetBlock):
        super(MSHNet,self).__init__()
        #添加heatmap+x+ResNet
        self.directionLayerList=[DirectionAssigned(i) for i in range(1,9)]
        self.conv_init=BasicLKBlock(in_channels,8,3,1,1)
        
        
        param_channels=[16,32,64,128,256]
        param_blocks_num=[2,2,2,2]
        self.pool=nn.MaxPool2d(2,2)
        self.up_2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up_4=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.up_8=nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.up_16=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)

        self.conv_start=nn.Conv2d(8,param_channels[0],1,1)

        #enconder2decoder
        self.encoder_0=self.pileBlock(param_channels[0],param_channels[0],block)
        self.encoder_1=self.pileBlock(param_channels[0],param_channels[1],block,blockNumber=2)
        self.encoder_2=self.pileBlock(param_channels[1],param_channels[2],block,blockNumber=2)
        self.encoder_3=self.pileBlock(param_channels[2],param_channels[3],block,blockNumber=2)

        self.mid_layer=self.pileBlock(param_channels[3],param_channels[4],block,blockNumber=2)

        self.decoder_3=self.pileBlock(param_channels[3]+param_channels[4],param_channels[3],block,blockNumber=2)
        self.decoder_2=self.pileBlock(param_channels[3]+param_channels[2],param_channels[2],block,blockNumber=2)
        self.decoder_1=self.pileBlock(param_channels[2]+param_channels[1],param_channels[1],block,blockNumber=2)
        self.decoder_0=self.pileBlock(param_channels[1]+param_channels[0],param_channels[0],block)
        
        self.output_0=nn.Conv2d(param_channels[0],1,1)
        self.output_1=nn.Conv2d(param_channels[1],1,1)
        self.output_2=nn.Conv2d(param_channels[2],1,1)
        self.output_3=nn.Conv2d(param_channels[3],1,1)

        self.final_output=nn.Conv2d(4,1,3,1,1)#torch.cat(mask*4) 通道数也为4
    def forward(self,x,tag):
        #添加heatmap+x
        directionResult=[self.directionLayerList[i](x) for i in range(len(self.directionLayerList))]
        directionRes=[directionResult[i].mul(directionResult[i+4]) for i in range(0,4)]
        
        directionUltraRes=directionRes[0]
        for i in range(1,len(directionRes)):
            directionUltraRes+=directionRes[i]
        directionUltraRes=nn.functional.sigmoid(directionUltraRes)
        out1=self.conv_init(x)
        out2=out1.mul(directionUltraRes)
        out=self.conv_start(out1+out2)
        
        
        en_0=self.encoder_0(out)
        en_1=self.encoder_1(self.pool(en_0))
        en_2=self.encoder_2(self.pool(en_1))
        en_3=self.encoder_3(self.pool(en_2))

        mid=self.mid_layer(self.pool(en_3))

        de_3=self.decoder_3(torch.cat([en_3,self.up_2(mid)],1))
        de_2=self.decoder_2(torch.cat([en_2,self.up_2(de_3)],1))
        de_1=self.decoder_1(torch.cat([en_1,self.up_2(de_2)],1))
        de_0=self.decoder_0(torch.cat([en_0,self.up_2(de_1)],1))

        if tag:
            mask_0=self.output_0(de_0)
            mask_1=self.output_1(de_1)
            mask_2=self.output_2(de_2)
            mask_3=self.output_3(de_3)
            result=torch.cat([mask_0,self.up_2(mask_1),self.up_4(mask_2),self.up_8(mask_3)],dim=1)
            result=self.final_output(result)

            return [mask_0,mask_1,mask_2,mask_3],result
        else:
            result=self.output_0(de_0)
            
            return [],result



    def pileBlock(self,in_channels,out_channels,block,blockNumber=1):
        netLayer=[]
        netLayer.append(block(in_channels,out_channels))
        for i in range(blockNumber-1):
            netLayer.append(block(out_channels,out_channels))
        return nn.Sequential(*netLayer)