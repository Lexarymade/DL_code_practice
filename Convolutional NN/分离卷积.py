import torch
import torch.nn as nn

#定义一个分离卷积的类
class separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(separable_conv, self).__init__()
        ## nin输入通道数，nout输出通道数
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=0, groups=nin)
        # groups(分组)要把原始的输入分成多少组，再这些组上分别卷积，最后返回拼接concat的结果
        # 深度无关卷积, 对每一个channel（二维）分别卷积，输出通道保持与输入通道数一致
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1) # nout个 nin * 1 * 1维度的卷积核

    def forward(self, x):
        out = self.depthwise(x)
        print(f"深度无关卷积后的大小是{out.size()}")
        out = self.pointwise(out)
        return out

if __name__ == '__main__':
    x = torch.ones([1, 8, 5, 5])
    sep_conv = separable_conv(8, 4)
    y = sep_conv(x)
    # with torch.no_grad():
    #     sep_conv.pointwise.weight[:,:,:,:] = torch.ones(4, 8, 1, 1) #纯娱乐，随便设置卷积层权重
    print(y.size())