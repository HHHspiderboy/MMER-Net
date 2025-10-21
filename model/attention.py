import torch.nn as nn
import torch


#注意力机制  maxpool+MLP  avgpool+MLP
class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),-1)
class ChannelAttention(nn.Module):
    def __init__(self,in_channels,reduction_ratio=16) :
        super(ChannelAttention,self).__init__()
        #定义MLP
        self.MLP=nn.Sequential(
            #展平
            Flatten(),
            nn.Linear(in_channels,in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio,in_channels)
        )
        
    def forward(self,x):
        avg_output=nn.functional.avg_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
        avg_output=self.MLP(avg_output)

        max_output=nn.functional.max_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
        max_output=self.MLP(max_output)
        
        
        scale=nn.functional.sigmoid(avg_output+max_output).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x*scale
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.compress=ChannelPool()
        self.spatial=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3,dilation=1,bias=False,groups=1),
            nn.BatchNorm2d(1,eps=1e-5,momentum=0.01,affine=True)
        )
    def forward(self,input):
        output=self.spatial(self.compress(input))
        output=nn.functional.sigmoid(output)
        return input*output

class AttentionSum(nn.Module):
    def __init__(self,channels,reduction_ratio=16):
        super(AttentionSum,self).__init__()
        self.channelAttention=ChannelAttention(channels,reduction_ratio)
        self.spatialAttention=SpatialAttention()
    def forward(self,input):
        out=self.spatialAttention(self.channelAttention(input))
        return out
    
class SGEAttention(nn.Module):
    def __init__(self, groups=8):
        super(SGEAttention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Group features and compute statistics
        x_grouped = x.view(batch_size * self.groups, channels_per_group, height, width)
        xn = x_grouped * self.avg_pool(x_grouped)
        xn = xn.sum(dim=1, keepdim=True)
        xn = xn.view(batch_size * self.groups, -1)
        
        # Normalize features
        xn_mean = xn.mean(dim=1, keepdim=True)
        xn_std = xn.std(dim=1, keepdim=True)
        xn = (xn - xn_mean) / (xn_std + 1e-5)
        xn = xn.view(batch_size, self.groups, height, width)
        
        # Generate attention weights
        xn = xn * self.weight + self.bias
        xn = xn.view(batch_size * self.groups, 1, height, width)
        xn = self.sig(xn)
        
        # Apply group-wise attention
        x_grouped = x_grouped * xn
        return x_grouped.view(batch_size, channels, height, width)   
    
#add new attention for local information_7.19
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=32):
        super(CoordinateAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.inter_channels = max(8, in_channels // reduction_ratio)
        
        # Convolution layers for feature transformation
        self.conv1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.conv_h = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        batch_size, _, height, width = x.size()
        
        # Coordinate information embedding
        x_h = F.adaptive_avg_pool2d(x, (height, 1))       # (N, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, width)).permute(0, 1, 3, 2)  # (N, C, W, 1)
        
        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)                  # (N, C, H+W, 1)
        y = F.relu(self.bn1(self.conv1(y)))                # (N, inter_C, H+W, 1)
        
        # Split and process height/width separately
        h_split, w_split = torch.split(y, [height, width], dim=2)
        w_split = w_split.permute(0, 1, 3, 2)              # (N, inter_C, 1, W)
        
        # Attention maps for height and width
        attn_h = torch.sigmoid(self.conv_h(h_split))        # (N, C, H, 1)
        attn_w = torch.sigmoid(self.conv_w(w_split))        # (N, C, 1, W)
        
        # Apply attention to original features
        return identity * attn_h.expand_as(x) * attn_w.expand_as(x)

class DPFA(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=4):
        super().__init__()
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, num_groups*2*3, 3, padding=1),
            nn.ReLU()
        )
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, groups=num_groups)
        
        # 位置注意力
        self.position_attn = nn.Sequential(
            nn.Conv2d(2, out_channels//8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels//8, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 生成位置网格
        b, c, h, w = x.shape
        pos_grid = self._create_position_grid(h, w).to(x.device)
        
        # 预测偏移量
        offset_mask = self.offset_predictor(x)
        offsets = offset_mask[:, :2*3*3]  # 3x3核的偏移量
        mask = torch.sigmoid(offset_mask[:, 2*3*3:])
        
        # 可变形卷积
        deform_feat = self.deform_conv(x, offsets, mask)
        
        # 位置注意力加权
        pos_attn = self.position_attn(pos_grid)
        return deform_feat * pos_attn
    
    def _create_position_grid(self, h, w):
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)