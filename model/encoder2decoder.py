'''
Large-Kernel Encoder:
x->3x3 Stem
Shape-Guided Decoder:

'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from .attention import *
class StemBlockUnit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(StemBlockUnit,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        out=self.net(x)
        return out

#3x3 Stem 16
class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock,self).__init__()
        self.unit1=StemBlockUnit(3,4)
        self.unit2=StemBlockUnit(4,8)
        self.conv1=nn.Conv2d(8,16,3,padding=1)
        self.BN=nn.BatchNorm2d(16)

    def forward(self,x):
        out=self.unit2(self.unit1(x))
        out=self.BN(self.conv1(out))
        return out
    
#KxK ResNet Cn
class ResNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResNetBlock,self).__init__()

        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.attention=AttentionSum(out_channels)
        self.ReLU=nn.ReLU(inplace=True)
        if in_channels!=out_channels :
            self.res_trans=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_trans=None
    def forward(self,x):
        
        if self.res_trans!=None:
            residual=self.res_trans(x)
        else:
            residual=x
        out=self.net(x)
        out=self.attention(out)
        out+=residual 
        out=self.ReLU(out)
        return out


#decoder:shape fusor part(torch.matmul())
class ShapeFusor(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ShapeFusor,self).__init__()
        #Low-level extractor
        self.low_feature_extractor=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//4,kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.Conv2d(out_channels//4,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        #High-level extractor
        self.high_feature_extractor_1=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.high_feature_extractor_2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels//4,kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.Conv2d(out_channels//4,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.high_feature_extractor_3=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Sigmoid()
        )
    def forward(self,x):
        low_out=self.low_feature_extractor(x)
        high_out_1=self.high_feature_extractor_1(low_out)
        high_out_2=self.high_feature_extractor_2(low_out)
        high_out_3=self.high_feature_extractor_3(low_out)
        temp_high_out_1=torch.matmul(high_out_1,high_out_2)
        temp_high_out_2=torch.matmul(high_out_2,high_out_3)
        out=torch.add(temp_high_out_1,temp_high_out_2)
        return out

# 3x3 Gated Conv
class GatedConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GatedConv,self).__init__()
        self.net1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.BN=nn.BatchNorm2d(out_channels)
        self.attention=AttentionSum(out_channels)
        
    def forward(self,x):
        out1=self.net1(x)
        out2=self.conv1(x)
        out=torch.matmul(out1,out2)
        out=self.BN(out)
        out=self.attention(out)
        return out




#head1 & head2 相同结构
class Head(nn.Module):
    def __init__(self,in_channels):
        super(Head,self).__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(in_channels,in_channels,3,1,1,0),
            nn.Conv2d(in_channels,4,kernel_size=3,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(4,1,kernel_size=1)
        )
    def forward(self,x):
        out=self.net(x)
        return out
#gated ResNet    
class GatedResNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GatedResNet,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        )
        self.gatedUnit=GatedConv(out_channels,out_channels)
        if in_channels!=out_channels:

            self.res_trans=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_trans=None
    def forward(self,encoder_out,edge_input):
        residual=edge_input
        if self.res_trans!=None:
            residual=self.res_trans(residual)
        edge_out=self.net(edge_input)
        temp_res=torch.add(encoder_out,edge_out)
        
        temp_sum=torch.add(temp_res,residual)

        gated_conv_output=self.gatedUnit(temp_sum)
        res_output=torch.add(torch.subtract(temp_sum,residual),temp_sum)
        out=torch.add(gated_conv_output,res_output)
        return out



    
class NetUpsampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(NetUpsampling,self).__init__()
        #修改upconv为3x3
        self.upConv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.resnet1=ResNetBlock(out_channels,out_channels,kernel_size=3)
        self.shapefusor_L=ShapeFusor(out_channels,out_channels)
        self.shapefusor_H=ShapeFusor(in_channels*2,out_channels)
    def forward(self,L,H):
       
        out=self.upConv(L)
        
        out=self.resnet1(out)
        
        out=self.shapefusor_L(out)
        H_out=self.shapefusor_H(H)
        out=torch.add(out,H_out)
        return out

#TFD-Edge Block
class TylorBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(TylorBlock,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),

        )
        self.relu=nn.ReLU(inplace=True)
        if in_channels!=out_channels or stride!=1:
            self.trans=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.trans=None
    def forward(self,x):
        residual=x
        
        out=self.net(x)
        
        if self.trans!=None:
            residual=self.trans(residual)
        out1=residual+out
        out1=self.relu(out1)
        out=self.relu(out)
        return out1,out

class TFDEdgeBlock(nn.Module):
    def __init__(self,in_channels,out_channles):
        super(TFDEdgeBlock,self).__init__()
        self.net1=TylorBlock(in_channels,out_channles)
        self.net2=TylorBlock(out_channles,out_channles)
        self.net3=GatedResNet(out_channles,out_channles)


    def forward(self,x,f_x):
        x_0=x
        x_1,delta_x_0=self.net1(x_0)
        _,x_2=self.net2(x_1)
        x_3_pre=self.net3(x_2,f_x)
        x_3=3*delta_x_0+x_2+x_3_pre
        return x_3



class EnhancedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EnhancedResNetBlock, self).__init__()
        
        # Convolutional layers
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.attention = CoordinateAttention(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.res_trans = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_trans = None

    def forward(self, x):
        residual = self.res_trans(x) if self.res_trans is not None else x
        out = self.net(x)
        out = self.attention(out)  # Apply selected attention
        out += residual
        return self.ReLU(out)
    
class ReducedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReducedResNetBlock, self).__init__()
        
        # Convolutional layers
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # self.attention = SGEAttention(groups=min(8, out_channels//8))
        self.ReLU = nn.ReLU(inplace=True)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.res_trans = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_trans = None

    def forward(self, x):
        residual = self.res_trans(x) if self.res_trans is not None else x
        out = self.net(x)
        # out = self.attention(out)  # Apply selected attention
        out += residual
        return self.ReLU(out)