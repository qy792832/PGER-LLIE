#############新加的
# models/llmoe.py
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
    def forward(self, x): return self.block(x)

class TimeEmbed(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, ch), nn.GELU(), nn.Linear(ch, ch))
    def forward(self, t_scalar):
        t = t_scalar.view(-1, 1)  # (B,1)
        return self.mlp(t)        # (B,ch)

class CEG(nn.Module):
    def __init__(self, ch, n_expert=4, topk=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(ch) for _ in range(n_expert)])
        self.router_f = nn.Conv2d(ch, n_expert, 1)   # sample-adaptive
        self.router_t = TimeEmbed(n_expert)          # time-adaptive
        self.topk = topk

    def forward(self, feat, t_scalar):
        B, C, H, W = feat.shape
        # 路由分数 = 图像路由 + 时间路由
        s_f = self.router_f(feat).mean(dim=(2,3))    # (B, E)
        s_t = self.router_t(t_scalar)                # (B, E)
        score = s_f + s_t
        # 稀疏 top-k
        topk_val, topk_idx = torch.topk(score, k=self.topk, dim=1)
        gate = torch.zeros_like(score).scatter(1, topk_idx, topk_val)
        gate = torch.softmax(gate, dim=1)            # (B, E)

        # 专家加权
        out = 0
        balance_loss = 0
        for e_idx, expert in enumerate(self.experts):
            w = gate[:, e_idx].view(B,1,1,1)
            out = out + w * expert(feat)
        # 负熵鼓励均衡（可选）
        p = gate.mean(dim=0)
        balance_loss = (p * (p.clamp_min(1e-6).log())).sum()
        return out, {'moe_balance': balance_loss, 'moe_sparsity': (gate**2).mean()}