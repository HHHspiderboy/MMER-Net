#去掉重建部分  
import torch.nn as nn
from .direction_metrics import DirectionAssigned
from .encoder2decoder import *
import torch
from .Sobel_detection import Sobel
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



    
class DERNet_ABA_v1(nn.Module):
    def __init__(self,in_channels,block=ResNetBlock):
        super(DERNet_ABA_v1,self).__init__()
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
        self.node1_0=self.pileBlock(param_channels[0],param_channels[1],block)
        self.node2_0=self.pileBlock(param_channels[1],param_channels[2],block)
        self.node3_0=self.pileBlock(param_channels[2],param_channels[3],block)
        self.node4_0=self.pileBlock(param_channels[3],param_channels[4],block)

        self.node0_1=self.pileBlock(param_channels[0],param_channels[0],block)
        self.node1_1=self.pileBlock(param_channels[0]+param_channels[1],param_channels[1],block,2)
        self.node2_1=self.pileBlock(param_channels[1]+param_channels[2],param_channels[2],block,2)
        self.node3_1=self.pileBlock(param_channels[2]+param_channels[3],param_channels[3],block,2)
        self.node4_1=self.pileBlock(param_channels[3]+param_channels[4],param_channels[4],block,2)

        self.node1_2=self.pileBlock(param_channels[1]+param_channels[2],param_channels[1],block,2)
        self.node2_2=self.pileBlock(param_channels[2]+param_channels[3],param_channels[2],block,2)
        self.node3_2=self.pileBlock(param_channels[3]+param_channels[4],param_channels[3],block,2)
        
        self.node0_2=self.pileBlock(param_channels[0]+param_channels[1]*2,param_channels[0],block)
        self.node1_3=self.pileBlock(param_channels[1]+param_channels[2],param_channels[1],block,2)
        self.node2_3=self.pileBlock(param_channels[2]+param_channels[3],param_channels[2],block,2)
        self.node3_3=self.pileBlock(param_channels[3]+param_channels[4],param_channels[3],block,2)
        self.node4_2=self.pileBlock(param_channels[4],param_channels[4],block,2)

        self.output_4=nn.Conv2d(param_channels[4],1,1)
        self.output_3=nn.Conv2d(param_channels[3],1,1)
        self.output_2=nn.Conv2d(param_channels[2],1,1)
        self.output_1=nn.Conv2d(param_channels[1],1,1)
        self.output_0=nn.Conv2d(param_channels[0],1,1)

        self.output=nn.Conv2d(param_channels[0],1,1)
        
        self.final_output=nn.Conv2d(5,1,3,1,1)

        #edge_map
        # self.Sobel=Sobel()
        # self.edge_init=nn.Conv2d(1,param_channels[0],1)
        # self.gatedRes_1=GatedResNet(param_channels[0],param_channels[4])#encoder->edge
        # # self.gatedRes_2=GatedResNet(param_channels[4],param_channels[4])
        # self.gatedRes_2=GatedResNet(param_channels[4],param_channels[3])
        # self.gatedRes_3=GatedResNet(param_channels[3],param_channels[2])
        # self.gatedRes_4=GatedResNet(param_channels[2],param_channels[1])
        # self.head=Head(param_channels[1])


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
        
        out0_0=self.node0_0(out)
        out1_0=self.node1_0(self.pool(out0_0))
        out2_0=self.node2_0(self.pool(out1_0))
        out3_0=self.node3_0(self.pool(out2_0))
        out4_0=self.node4_0(self.pool(out3_0))

        out0_1=self.node0_1(out0_0)
        out1_1=self.node1_1(torch.cat([self.down(out0_1),out1_0],dim=1))
        out2_1=self.node2_1(torch.cat([self.down(out1_1),out2_0],dim=1))
        out3_1=self.node3_1(torch.cat([self.down(out2_1),out3_0],dim=1))
        out4_1=self.node4_1(torch.cat([self.down(out3_1),out4_0],dim=1))

        
        

        out3_2=self.node3_2(torch.cat([out3_1,self.up_2(out4_1)],dim=1))
        out2_2=self.node2_2(torch.cat([out2_1,self.up_2(out3_2)],dim=1))
        out1_2=self.node1_2(torch.cat([out1_1,self.up_2(out2_2)],dim=1))
        


        out4_2=self.node4_2(out4_1)
        out3_3=self.node3_3(torch.cat([out3_2,self.up_2(out4_2)],dim=1))
        out2_3=self.node2_3(torch.cat([out2_2,self.up_2(out3_3)],dim=1))
        out1_3=self.node1_3(torch.cat([out1_2,self.up_2(out2_3)],dim=1))
        out0_2=self.node0_2(torch.cat([out0_1,self.up_2(out1_2),self.up_2(out1_3)],dim=1))
        
        



        if tag:
            mask4=self.output_4(out4_2)
            mask3=self.output_3(out3_3)
            mask2=self.output_2(out2_3)
            mask1=self.output_1(out1_3)
            mask0=self.output_0(out0_2)
            result=torch.cat([mask0,self.up_2(mask1),self.up_4(mask2),self.up_8(mask3),self.up_16(mask4)],dim=1)
            result=self.final_output(result)
            # if edge_tag:
            #     #过了warm_epoch再进行edge_map的训练
            #     edge_out=self.Sobel(x)
            #     edge_out=self.edge_init(edge_out)
            #     edge_out=self.gatedRes_1(self.up_16(out4_1),edge_out)
            #     # edge_out=self.gatedRes_2(self.up_16(out4_2),edge_out)
            #     edge_out=self.gatedRes_2(self.up_8(out3_3),edge_out)
            #     edge_out=self.gatedRes_3(self.up_4(out2_3),edge_out)
            #     edge_out=self.gatedRes_4(self.up_2(out1_3),edge_out)
            #     edge_out=self.head(edge_out)

            #     return [mask0,mask1,mask2,mask3,mask4],result
            # else:
            return [mask0,mask1,mask2,mask3,mask4],result
            

        else:
            
            result=self.output(out0_2)
            

            return [],result



        

    def pileBlock(self,in_channels,out_channels,block,blockNumber=1):
        netLayer=[]
        netLayer.append(block(in_channels,out_channels))
        for i in range(blockNumber-1):
            netLayer.append(block(out_channels,out_channels))
        return nn.Sequential(*netLayer)