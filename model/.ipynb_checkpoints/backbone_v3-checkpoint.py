import torch.nn as nn
from .direction_metrics import DirectionAssigned
from .encoder2decoder import *
import torch
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
    
#1.仅第一/二层能够接收来自深层节点的feature map传输
#2.考虑是否去掉最开始的一个conv
class MultiNet_v3(nn.Module):
    def __init__(self,in_channels,block=ResNetBlock):
        super().__init__()
        param_channels=[16,32,64,128,256]

        self.directionLayerList=[DirectionAssigned(i) for i in range(1,9)]
        self.conv_init=BasicLKBlock(in_channels,8,3,1,1)
        self.conv_start=nn.Conv2d(8,param_channels[0],1,1)

        self.pool=nn.MaxPool2d(2,2)
        self.ReLU=nn.ReLU(inplace=True)
        self.down=nn.Upsample(scale_factor=0.5,mode='bilinear',align_corners=True)
        self.up_2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up_4=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.up_8=nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.up_16=nn.Upsample(scale_factor=16,mode='bilinear',align_corners=True)

        self.node0_0=self.pileBlock(param_channels[0],param_channels[0],block)
        self.node1_0=self.pileBlock(param_channels[0],param_channels[1],block,2)
        self.node2_0=self.pileBlock(param_channels[1],param_channels[2],block,2)
        self.node3_0=self.pileBlock(param_channels[2],param_channels[3],block,2)
        self.node4_0=self.pileBlock(param_channels[3],param_channels[4],block,2)

        self.node0_1=self.pileBlock(param_channels[0]+param_channels[1],param_channels[0],block)
        self.node1_1=self.pileBlock(param_channels[1]+param_channels[2]+param_channels[0],param_channels[1],block,2)
        self.node2_1=self.pileBlock(param_channels[2]+param_channels[3]+param_channels[1],param_channels[2],block,2)
        self.node3_1=self.pileBlock(param_channels[3]+param_channels[2]+param_channels[4],param_channels[3],block,2)

        self.node0_2=self.pileBlock(param_channels[0]*2+param_channels[1],param_channels[0],block)
        self.node1_2=self.pileBlock(param_channels[0]+param_channels[1]*2+param_channels[2],param_channels[1],block,2)
        self.node2_2=self.pileBlock(param_channels[1]+param_channels[2]*2+param_channels[3],param_channels[2],block,2)

        self.node0_3=self.pileBlock(param_channels[0]*3+param_channels[1],param_channels[0],block)
        self.node1_3=self.pileBlock(param_channels[1]*3+param_channels[0]+param_channels[2],param_channels[1],block,2)

        self.node0_4=self.pileBlock(param_channels[0]*4+param_channels[1],param_channels[0],block)

        self.out_node=self.pileBlock(param_channels[0]*5,param_channels[0],block)
        #4 Conv
        self.conv_1=nn.Conv2d(param_channels[4],param_channels[0],1,1)
        self.conv_2=nn.Conv2d(param_channels[3],param_channels[0],1,1)
        self.conv_3=nn.Conv2d(param_channels[2],param_channels[0],1,1)
        self.conv_4=nn.Conv2d(param_channels[1],param_channels[0],1,1)
        #output conv 1x1
        self.output_3=nn.Conv2d(param_channels[0],1,1)
        self.output_2=nn.Conv2d(param_channels[0],1,1)
        self.output_1=nn.Conv2d(param_channels[0],1,1)
        self.output_0=nn.Conv2d(param_channels[0],1,1)
        
        self.output=nn.Conv2d(param_channels[0],1,1)
        self.final=nn.Conv2d(4,1,3,1,1)


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
        #第一层
        out0_0=self.node0_0(out)
        out1_0=self.node1_0(self.pool(out0_0))
        out0_1=self.node0_1(torch.cat([out0_0,self.up_2(out1_0)],dim=1))
        
        #第二层
        out2_0=self.node2_0(self.pool(out1_0))
        out1_1=self.node1_1(torch.cat([out1_0,self.up_2(out2_0),self.down(out0_1)],dim=1))
        out0_2=self.node0_2(torch.cat([out0_0,out0_1,self.up_2(out1_1)],dim=1))
        
        #第三层
        out3_0=self.node3_0(self.pool(out2_0))
        out2_1=self.node2_1(torch.cat([out2_0,self.up_2(out3_0),self.down(out1_1)],dim=1))
        out1_2=self.node1_2(torch.cat([out1_0,out1_1,self.up_2(out2_1),self.down(out0_2)],dim=1))
        out0_3=self.node0_3(torch.cat([out0_0,out0_1,out0_2,self.up_2(out1_2)],dim=1))

        #第四层
        out4_0=self.node4_0(self.pool(out3_0))
        out3_1=self.node3_1(torch.cat([out3_0,self.up_2(out4_0),self.down(out2_1)],dim=1))
        out2_2=self.node2_2(torch.cat([out2_0,out2_1,self.up_2(out3_1),self.down(out1_2)],dim=1))
        out1_3=self.node1_3(torch.cat([out1_0,out1_1,out1_2,self.up_2(out2_2),self.down(out0_3)],dim=1))
        out0_4=self.node0_4(torch.cat([out0_0,out0_1,out0_2,out0_3,self.up_2(out1_3)],dim=1))
       
        #做上采样将五个最外层的进行连接
        result=self.out_node(torch.cat([
            self.up_16(self.conv_1(out4_0)),self.up_8(self.conv_2(out3_1)),
            self.up_4(self.conv_3(out2_2)),self.up_2(self.conv_4(out1_3)),out0_4],dim=1))
        
        if tag:
            mask0=self.output_0(out0_1)
            mask1=self.output_1(out0_2)
            mask2=self.output_2(out0_3)
            mask3=self.output_3(result)
            # result=torch.cat([mask0,mask1,mask2,mask3],dim=1)
            # result=self.final(result)

            return [mask0,mask1,mask2],mask3
        else:
            result=self.output(result)

            return [],result


    def pileBlock(self,in_channels,out_channels,block,blockNumber=1):
        netLayer=[]
        netLayer.append(block(in_channels,out_channels))
        for i in range(blockNumber-1):
            netLayer.append(block(out_channels,out_channels))
        return nn.Sequential(*netLayer)