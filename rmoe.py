##########新加的
# models/rmoe.py
import torch, torch.nn as nn, torch.nn.functional as F
from models.decom import CTDN
from models.rsam import RSAM
from models.llmoe import LLMoE

class RestoreWithRMoE(nn.Module):
    def __init__(self, ch=64, use_teacher=True):
        super().__init__()
        self.ctdn = CTDN(ch)
        # 轻量编码/解码（你也可以直接复用 UNet 的某层特征）
        self.enc = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.GELU())
        self.dec = nn.Sequential(nn.Conv2d(ch, 3, 3, 1, 1))
        self.rsam = RSAM(ch)
        self.moe  = LLMoE(ch, n_expert=4, topk=2)
        self.use_teacher = use_teacher

    @torch.no_grad()
    def teacher_decompose(self, img):
        # 可选：对“正常光图”（数据集中第二张）或 x0 使用 CTDN 做教师分解
        # 这里演示对 img 自身做分解；也可以在外部传入 normal 图
        pair = torch.cat([img, img], dim=1)  # 伪对；若有正常光图请用 [low, high]
        out = self.ctdn(pair, pred_fea=None)
        return out["high_R"], out["high_L"]

    def forward(self, low_img_01, x0_01, t_scalar):
        """
        low_img_01: [B,3,H,W], 0-1
        x0_01:      [B,3,H,W], 0-1
        t_scalar:   [B,1] 或 [B]，时间步/噪声级
        """
        # 1) 用 CTDN 取潜空间监督（这里把 x0 当“高光分支”喂给 CTDN）
        pair = torch.cat([low_img_01, x0_01], dim=1)           # [B,6,H,W]
        out = self.ctdn(pair, pred_fea=None)
        teacher_R, teacher_L = out["high_R"], out["high_L"]    # 教师

        # 2) 编码 x0 为特征，走 R-SAM
        feat = self.enc(x0_01)
        feat, loss_rsam, (predR, predL) = self.rsam(feat, teacher_R, teacher_L)

        # 3) 走 LLMoE
        feat, loss_moe = self.moe(feat, t_scalar.float())

        # 4) 解码得到增强结果
        x_out = torch.clamp(self.dec(feat), 0, 1)

        # 5) 收集监督量（与你现有的 supervised_* 命名保持一致即可）
        supervised_l_list = [predL, teacher_L]
        supervised_r_list = [predR, teacher_R]
        l_MoE = loss_moe['moe_balance'] + 1e-3 * loss_moe['moe_sparsity']
        l_RSAM = loss_rsam['rsam_R'] + 0.3 * loss_rsam['rsam_L']
        return x_out, supervised_l_list, supervised_r_list, (l_MoE + l_RSAM)
    
###说明：LightenDiffusion 的 restore_fn 预期返回 (x_out, supervised_l_list, supervised_r_list, l_MoE) 这样的结构（你现有代码就这么用的），上面的适配器完全对齐这个协议