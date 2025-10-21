import torch
import torch.nn as nn

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 Sobel 核（水平x和垂直y方向）
        self.kernel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)  # 形状 [out_ch, in_ch, H, W]
        
        self.kernel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 固定卷积核参数（不参与梯度更新）
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight.data = self.kernel_x
        self.conv_y.weight.data = self.kernel_y
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        if img.size(1) == 3:
            img = torch.mean(img, dim=1, keepdim=True)  # RGB转灰度
            
        # 计算水平和垂直梯度
        grad_x = self.conv_x(img)
        grad_y = self.conv_y(img)
        
        # 计算梯度幅值 (等效于 OpenCV 的 addWeighted)
        edge = torch.sqrt(grad_x**2 + grad_y**2)  # 绝对值通过平方和平方根处理
        
        # 归一化到 [0,1]
        edge_map = edge / edge.max()  # 或固定归一化 edge_map = edge / 255.0
        
        return edge_map