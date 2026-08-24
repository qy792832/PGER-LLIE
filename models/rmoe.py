# models/rsam.py
###################新加的
import torch, torch.nn as nn, torch.nn.functional as F

class LatentRetinexDecoder(nn.Module):## --- 新名字：RetinexPriorHead（旧 LatentRetinexDecoder） ---
    def __init__(self, ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
        )
        self.to_R = nn.Conv2d(ch, 3, 1)  # 预测 R (3通道)
        self.to_L = nn.Conv2d(ch, 1, 1)  # 预测 L (1通道)

    def forward(self, f):
        h = self.head(f)
        R = torch.sigmoid(self.to_R(h))
        L = torch.sigmoid(self.to_L(h))
        L = L.repeat(1, 3, 1, 1)  # 对齐到3通道
        return R, L

class RSAM(nn.Module):## --- 外层模块：PGFM（可额外提供 FeatureModulator 别名） ---
    def __init__(self, ch):
        super().__init__()
        self.lrd = LatentRetinexDecoder(ch)
        self.attn = nn.Sequential(
            nn.Conv2d(ch, ch, 1), nn.Sigmoid()
        )

    def forward(self, feat, teacher_R, teacher_L):
        # 1) 预测 Retinex
        pred_R, pred_L = self.lrd(feat)
        # 2) 蒸馏监督（教师分解必须 stop-grad）
        loss_R = F.mse_loss(pred_R, teacher_R.detach())
        loss_L = F.mse_loss(pred_L, teacher_L.detach())
        # 3) 用注意力调制特征
        a = self.attn(feat)
        feat_out = feat * (1 + a)
        return feat_out, {'rsam_R': loss_R, 'rsam_L': loss_L}, (pred_R, pred_L)
###论文里的 R-SAM 是“多尺度 + 停梯度蒸馏”的思想，以上是最小实现：轻量两层卷积做 LRD + 一个通道注意力调制即可。后续可扩成金字塔多尺度
