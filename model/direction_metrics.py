import torch.nn as nn
import numpy as np
import math
import torch

#计算direction位置
def direction_compute(direction):
    #代表8个不同的方向
    #1(0,0) 2(0,2) 3(0,4) 4(2,4) 5(4,4) 6(4,2) 7(4,0) 8(2,0)
    coodinates={
        1:(0,0),2:(0,2),3:(0,4),4:(2,4),
        5:(4,4),6:(4,2),7:(4,0),8:(2,0)
    }
    return coodinates[direction]

class DirectionAssigned(nn.Module):
    def __init__(self,direction):
        super(DirectionAssigned,self).__init__()
        kernel=np.zeros((5,5))
        kernel[2][2]=1 #中心点
        x,y=direction_compute(direction)
        kernel[x][y]=-1
        #转换维度至四维
        kernel=torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel=kernel.to('cuda')#服务器跑需要统一类型
        self.weight=nn.Parameter(data=kernel,requires_grad=False)
        
        
    def forward(self,x):
        return nn.functional.conv2d(x,self.weight,padding=2)
        