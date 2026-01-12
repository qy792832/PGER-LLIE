import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Depth_conv(nn.Module): #定义一个继承自 torch.nn.Module 的类，名字叫 Depth_conv，用于实现“深度可分离卷积”
    def __init__(self, in_ch, out_ch): #输入通道数和输出通道数
        super(Depth_conv, self).__init__() #super(Depth_conv, self):这表示：获取 Depth_conv 类的父类，也就是 nn.Module;   . __init__():这表示：调用父类的构造函数（也就是 nn.Module.__init__()）
        self.depth_conv = nn.Conv2d( #深度卷积层
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch #每个输入通道对应一个独立的卷积核，不跨通道计算；因此这里实际上是进行了 in_ch 个 3×3 卷积操作，每个卷积处理一个通道
        )
        self.point_conv = nn.Conv2d( #逐点卷积层
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1), #1x1 卷积，作用是对“各个通道”进行融合与映射到 out_ch 个输出通道；这是深度可分离卷积的“通道混合”阶段
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input) #对输入的每个通道单独做 3×3 卷积
        out = self.point_conv(out) #用 1×1 卷积将这些特征通道混合，输出你想要的 out_ch 个通道
        return out #返回这个结果


class Res_block(nn.Module): #这是定义了一个类 Res_block，继承自 torch.nn.Module，表示要自定义一个神经网络模块
    def __init__(self, in_channels, out_channels): #输入通道数和输出通道数
        super(Res_block, self).__init__()#super(Res_block, self):这表示：获取 Res_block 类的父类，也就是 nn.Module;   . __init__():这表示：调用父类的构造函数（也就是 nn.Module.__init__()）

        sequence = []#创建一个空列表，用于存储神经网络中的层

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),# nn.Conv2d:卷积层 用于实现 二维卷积操作;对图像进行特征提取，通过滑动一个卷积核，对图像局部区域进行加权求和操作
            nn.LeakyReLU(),#激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]#创建一个名为 sequence 的列表，用于存储神经网络中的层

        self.model = nn.Sequential(*sequence)#创建一个名为 model 的神经网络，用于存储神经网络中的层
        #*sequence 这个写法是 Python 语法的一个重要特性，叫做 “参数解包（unpacking）;把列表 sequence 中的每个元素单独取出来，作为参数传给 nn.Sequential(...)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)#创建一个名为 conv 的卷积层，用于将输入的通道数映射到输出的通道数
        #这是残差连接的关键！有时候 in_channels ≠ out_channels，你不能直接把 x + self.model(x) 相加，会报错;所以我们加一个 1×1 卷积，来“调整通道数”
    def forward(self, x):
        out = self.model(x) + self.conv(x)#残差连接  self.model(x) 是对输入 x 进行一系列卷积和激活操作后的输出，而 self.conv(x) 则是对输入 x 进行 1×1 卷积后的输出

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()#继承父类 nn.Module

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

#它的作用是 “通道压缩+图像重建”，常用于将特征图（很多通道）恢复成图像（3 通道 RGB）。
class channel_down(nn.Module): #自定义了一个模块，继承自 nn.Module，命名为 “通道向下压缩模块”。
    def __init__(self, channels):
        super(channel_down, self).__init__()
        #定义了3层卷积层结构
        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        #输入：channels × 4（假如 channels=64，那就是 256 通道）；输出：channels × 2（128 通道）；卷积核大小是 3×3，padding=1 保持空间大小不变；功能：通道降一半
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)#输入：上一层输出 128 通道；输出：channels（64 通道）
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)#输入：64 通道；输出：3 通道（RGB 图像）；最后我们会用 sigmoid 把像素值压缩到 [0, 1] 范围

        self.relu = nn.LeakyReLU()#每一层卷积后都使用 LeakyReLU 激活函数，增强模型的非线性表示能力

    def forward(self, x):
        out = torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))

        return out

# 将图像的通道数从 3 通道（RGB）升高为更多的通道数（例如 64 → 128 → 256），用于特征提取或上游网络的恢复处理。
class channel_up(nn.Module):
    def __init__(self, channels):
        super(channel_up, self).__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))

        return out


class feature_pyramid(nn.Module):#从输入图像中提取不同层次的空间特征，形成 多尺度特征金字塔，用于后续的图像增强或重建处理
    def __init__(self, channels):
        super(feature_pyramid, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
                                   nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2))

        self.block0 = Res_block(channels, channels)#block0 是一个残差块（即特征增强）
        #down0 是一个下采样卷积层：高宽减半，通道不变;用 stride=2 做下采样;输出：level0（尺寸是输入的 1/2）
        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = Res_block(channels, channels * 2)#也就是说，这个残差块的输入通道是 channels，输出通道是 channels * 2。这是在升维！

        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = Res_block(channels * 2, channels * 4)

        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))

        return level0, level1, level2


class ReconNet(nn.Module):#这是整个CTDN 网络中的一个重要组成部分，用于提取多尺度特征 + 解码生成图像。 ReconNet 是一个金字塔式特征提取 + 解码器结构
    def __init__(self, channels): #传入的 channels 是基准通道数（比如 64），后续会按倍数扩大
        super(ReconNet, self).__init__() #继承父类 nn.Module

        self.pyramid = feature_pyramid(channels)#这是一个 U-Net 编码结构，提取三层不同分辨率的特征; level0：1/2 尺寸，通道 channels; level1：1/4 尺寸，通道 channels * 2

        self.channel_down = channel_down(channels) #将高通道压缩为 3 通道（用于解耦特征 → 图像）
        self.channel_up = channel_up(channels) #将 3 通道重新编码为高通道（用于恢复解码）

        self.block_up0 = Res_block(channels * 4, channels * 4)                  #残差解码器部分;以下部分是一个逐层上采样 + 残差融合结构：
        self.block_up1 = Res_block(channels * 4, channels * 4)                  #self.block_up0 ~ block_up5
        self.up_sampling0 = upsampling(channels * 4, channels * 2)              #self.up_sampling0 ~ up_sampling2
        self.block_up2 = Res_block(channels * 2, channels * 2)                  #每次上采样阶段结构为：
        self.block_up3 = Res_block(channels * 2, channels * 2)                  #block → block → upsample → 下一级融合
        self.up_sampling1 = upsampling(channels * 2, channels)
        self.block_up4 = Res_block(channels, channels)
        self.block_up5 = Res_block(channels, channels)
        self.up_sampling2 = upsampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=(1, 1), stride=(1, 1), padding=0) #最后将通道从 channels → 3，输出为 RGB 图像

        self.relu = nn.LeakyReLU()

    def forward(self, x, pred_fea=None):

        if pred_fea is None: #pred_fea is None:这表示模型处于 编码模式（提取低光 / 正常图像特征）
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])#self.pyramid 是一个 feature_pyramid 模块;他会返回3个值
            low_fea_down8 = self.channel_down(low_fea_down8)

            high_fea_down2, high_fea_down4, high_fea_down8 = self.pyramid(x[:, 3:, ...])
            high_fea_down8 = self.channel_down(high_fea_down8)
            #输入是一个 6 通道图像（拼接的低光 + 正常图像）；分别对前 3 通道（低光）和后 3 通道（正常）做 pyramid 提取；提取出第3层（最深）的特征 → channel_down → 得到 [B, 3, H/8, W/8] 形式的深层压缩特征
            return low_fea_down8, high_fea_down8#这两个将交给后续模块使用
        else:
            # =================low ori decoder================= pred_fea is not None：这是解码模式：输入低光图像 + 表达特征 pred_fea，恢复增强图像
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])
            #从低光图像中重新提取 pyramid 特征（作为 skip 连接用）；将 pred_fea 从 3 通道 → channels * 4（通过 channel_up）
            pred_fea = self.channel_up(pred_fea)

            pred_fea_up2 = self.up_sampling0(
                self.block_up1(self.block_up0(pred_fea) + low_fea_down8))       #从 pred_fea 开始，逐级上采样并融合 skip 连接特征
            pred_fea_up4 = self.up_sampling1(
                self.block_up3(self.block_up2(pred_fea_up2) + low_fea_down4))           #注意每一层都先：
            pred_fea_up8 = self.up_sampling2(                                           #残差 block ×2；加上对应层 low_fea_downX 的 skip 跳跃连接；上采样恢复分辨率
                self.block_up5(self.block_up4(pred_fea_up4) + low_fea_down2))

            pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))                  #通过两个卷积（3×3 + 1×1）将特征图变成 3 通道图像

            return pred_img                                                             #这是增强后的图像 pred_img


class Self_Attention(nn.Module):            #是一个基于卷积的 图像自注意力模块；作用：在图像中构建像素之间的全局关系（非局部交互），让每个位置能关注图像中其他区域的特征，从而提升图像理解和重建能力。
    def __init__(self, dim, num_heads, bias): #dim: 输入通道数（例如 64、128 等）；num_heads: 多头注意力的头数；bias: 是否给卷积加偏置
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads 
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias) #用 1×1 卷积把输入张量映射成 Query（Q）/Key（K）/Value（V）；输入 [B, C, H, W] → 输出 [B, 3C, H, W]
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),#这是一种Depthwise 卷积
                                    padding=1, groups=dim * 3, bias=bias)#对每个通道单独做 3x3 卷积（没有通道融合）；目的是：扩大 Q/K/V 的局部感受野；输出仍然是 [B, 3C, H, W]
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)#这个是最后一层，用于「输出映射」，维持通道不变
        #nn.Conv2d 是 PyTorch 中的二维卷积操作，是处理图像的基础模块。作用：在输入图像上滑动一个可学习的卷积核（filter），提取局部特征，生成新的特征图（feature map）。
    def forward(self, x):
        b, c, h, w = x.shape#b：batch size；c：通道数（即 dim）；h, w：图像高和宽

        qkv = self.qkv_dwconv(self.qkv(x)) #self.qkv(x) 得到 3C 通道张量（Q+K+V拼在一起）
        q, k, v = qkv.chunk(3, dim=1)#chunk(3, dim=1) 把它均分成三块：Q/K/V

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#是 einops 提供的 rearrange 函数在进行张量变形。具体怎么实现的，自己搜
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)#
        #也就是把原图展平成二维向量，为了后面做矩阵乘法。
        q = torch.nn.functional.normalize(q, dim=-1)#把每个 Query 和 Key 向量 L2 归一化
        k = torch.nn.functional.normalize(k, dim=-1)#这样后续的 q @ kᵀ 结果是余弦相似度
        #计算注意力图
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)#对每个像素点，拿 V 的加权和；得到所有注意力信息融合后的结果
        #还原形状 + 投影
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)#把多头合并回原通道维度（和输入 x 同形状）；再通过 1×1 卷积映射为最终输出

        out = self.project_out(out)
        return out#和输入一样大 ;但每个像素包含了全图上下文信息;比普通卷积更具“全局理解能力”


class Cross_Attention(nn.Module):#Self Attention：Q、K、V 都来自同一个输入（比如图像自身）;Cross Attention：Q 来自一个输入（比如待增强图），K 和 V 来自另一个输入（比如参考图）
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0: #dim：输入特征的通道数;num_heads：注意力头的数量;dim / num_heads：每个头的通道数，称为 attention_head_size，必须能整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        # 初始化卷积模块（用于构建 Q/K/V）
        self.query = Depth_conv(in_ch=dim, out_ch=dim)#这里使用了你前面定义的 深度可分离卷积（Depthwise Separable Convolution），提取 Q、K、V。
        self.key = Depth_conv(in_ch=dim, out_ch=dim)    #每个输出 shape 都是 [B, dim, H, W]
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)#在 attention 上添加 Dropout（默认是 0），防止过拟合

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)#把张量维度换个顺序，以适配后面 matmul 的形状要求

    def forward(self, hidden_states, ctx):#这里 forward() 接收两个输入； hidden_states：生成 Q，用于「提问」； ctx：生成 K、V，用于「提供上下文信息」
        #构建 Q、K、V；每个结果形状都是 [B, dim, H, W]，比如 [4, 64, 32, 32]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer) ##这里是预处理，为了后面做矩阵乘法
        key_layer = self.transpose_for_scores(mixed_key_layer)      #假设输入：B = 4, dim = 64, H = W = 32，num_heads = 8, 那每个 head 的通道就是 64/8 = 8
        value_layer = self.transpose_for_scores(mixed_value_layer)  #这些张量 shape 要变成 [B, H×W, num_heads, head_dim] 形式才能做注意力计算，通常操作是 reshape 成 [B, num_heads, head_dim, H×W]
        #计算 Attention Scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#计算 Q 和 K 的相似度：[B, heads, q_len, k_len],.transpose(-1, -2)：转置最后两个维度,这样才可以用 matmul 计算 dot product
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#除以根号 d，是标准的注意力缩放技巧，防止梯度过大

        attention_probs = nn.Softmax(dim=-1)(attention_scores)#把 raw 相似度转换为概率（权重总和为 1）

        attention_probs = self.dropout(attention_probs)#丢弃一部分注意力（训练时生效）
        #加权求和（得到上下文）
        ctx_layer = torch.matmul(attention_probs, value_layer)#用注意力概率对 V 做加权求和;结果是：[B, heads, head_dim, HW]
        #恢复形状
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer #Cross Attention 的作用:信息迁移：从另一个特征图（比如正常图）中提取有用信息;结构对齐：提升低光图在光照建模时的结构感知能力;特征增强：提高分解模块对复杂图像内容的理解能力


class Retinex_decom(nn.Module):#整个增强系统中Retinex分解模块的核心
    def __init__(self, channels):
        super(Retinex_decom, self).__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))#这两行是处理 反射率图的网络，输入是一个 RGB 三通道图像（比如归一化后的 $I / L$），输出是中间特征。

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))#这两行是处理 照明图的网络，输入是一通道图像（$L$），输出也是中间特征。

        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)#让照明图从反射图中提取结构信息
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)#让照明图自己学习其全局特征

        self.conv0_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1))#self.conv0_1 = ... → 输出 R
        self.conv1_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1))#self.conv1_1 = ... → 输出 L
        #使用 ResBlock + Conv2d 组合，将融合后的特征生成最终的反射率和照明图。
    def forward(self, x):
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]#这一步从原始图像 x 中估计出一个初始照明图;对于每个像素，取 RGB 三通道的最大值作为该位置的光照估计;输出形状是 [B, 1, H, W]
        init_reflectance = x / init_illumination#用原图除以初始照明，估计出一个初始反射率图

        Reflectance, Illumination = (self.blocks0(self.conv0(init_reflectance)),#你给初始 R, L 各通过一个特征提取网络（Conv + 2 个 ResBlock）
                                     self.blocks1(self.conv1(init_illumination)))#输出特征形状是 [B, C, H, W]（比如 C=64）

        Reflectance_final = self.cross_attention(Illumination, Reflectance)#让照明图（Illumination）“提问”反射图（Reflectance），获取结构信息; 这非常重要，因为：
        #L 主要表示光照强度，通常不含有结构纹理信息；通过 cross attention 融合R的结构信息，让 L更精准、更具有上下文意义
        Illumination_content = self.self_attention(Illumination)#这是让照明图自己学习全局特征，用于后续调整输出的细节（比如全局亮度平衡）。
        #融合处理输出
        Reflectance_final = self.conv0_1(Reflectance_final + Illumination_content)
        Illumination_final = self.conv1_1(Illumination - Illumination_content)
        #激活 & 输出拼接
        R = torch.sigmoid(Reflectance_final)
        L = torch.sigmoid(Illumination_final)#将两个图激活到 0-1 区间，保证值域合理（可视化或乘原图都能正常使用）
        L = torch.cat([L for i in range(3)], dim=1)#照明图原本是 [B, 1, H, W]，拼接三次 → [B, 3, H, W]，方便后续与原图相乘或送入网络

        return R, L#最终输出的是三通道的R,L反射和照明图；这就实现了一个完整的 Retinex 分解网络！


class CTDN(nn.Module):             #它整合了两个子模块：ReconNet：图像重建网络；      Retinex_decom：图像分解网络（提取反射率 R 和光照图 L）

    def __init__(self, channels=64):
        super(CTDN, self).__init__()

        self.ReconNet = ReconNet(channels)
        self.retinex = Retinex_decom(channels)#初始化了两个核心模块，ReconNet 用于图像特征提取、上采样、重建；Retinex_decom 用于对特征进行 Retinex 分解，输出 R（反射率）和 L（照明图）。

    def forward(self, images, pred_fea=None):#images：输入图像，形状一般是 [B, 6, H, W]（低光图 + 正常光图）； pred_fea：是否处于训练前阶段（图像解耦）还是重建阶段（生成增强图）

        output = {}
        # =================decomposition low=================
        if pred_fea is None: #第一种情况：pred_fea is None 时；说明当前是在 图像编码与分解阶段
            low_fea_down8, high_fea_down8 = self.ReconNet(images, pred_fea=None)
            #输入图像 images 是 拼接了低光图和正常图的 6 通道张量；  images[:, :3, ...] → 低光图；   images[:, 3:, ...] → 正常图
            low_R, low_L = self.retinex(low_fea_down8)
            high_R, high_L = self.retinex(high_fea_down8)
            #送入 ReconNet 之后，会分别提取这两个图的高层特征（8倍下采样后的表示）；  low_fea_down8：低光图的特征     ；high_fea_down8：正常图的特征
            output["low_R"] = low_R
            output["low_L"] = low_L
            output["low_fea"] = low_fea_down8
            output["high_R"] = high_R
            output["high_L"] = high_L
            output["high_fea"] = high_fea_down8
            #把所有中间结果保存在字典 output 中，方便训练使用：
        else:#pred_fea 不为 None 时;说明我们是在 图像重建阶段
            pred_img = self.ReconNet(images[:, :3, ...], pred_fea=pred_fea)#输入图像是低光图（3通道）;pred_fea 是我们从前面（比如高光图、L 变换后）传过来的重构特征
                                                                           #对生成的图像进行后处理（比如归一化、裁剪等），确保输出符合预期范围 
            #这个 output 字典是整个 CTDN 模块的输出：在特征分解阶段：输出 R, L, 特征图;在重建阶段：输出增强后的图像 pred_img
            output["pred_img"] = pred_img                                   
            
           
            
        return output
