import os
import time
import datetime
import math
from collections import deque
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import Image

import utils
from models.unet import DiffusionUNet
from models.decom import CTDN
from models.pgfm import PGFM
from models.ceg import CEG


def _save_ckpt_exact(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)  # 不做任何二次命名


# =========================
# 工具：图像/PSNR/EMA 验证上下文
# =========================
def _to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    """PIL -> Tensor [C,H,W], 0~1"""
    arr = np.array(img.convert('RGB'), dtype=np.uint8)  # H,W,3
    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return ten


@torch.no_grad()
def _load_gt_tensor(y_item: Union[str, os.PathLike], gt_root: str) -> torch.Tensor:
    """y_item 可能是文件名或相对路径；gt_root 通常是 val_dir/high 或 val_dir。"""
    cand = str(y_item)
    if not os.path.isabs(cand):
        cand = os.path.join(gt_root, cand)
    if not os.path.exists(cand):
        base, _ = os.path.splitext(os.path.basename(str(y_item)))
        exts = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']
        for e in exts:
            p = os.path.join(gt_root, base + e)
            if os.path.exists(p):
                cand = p
                break
    img = Image.open(cand).convert('RGB')
    return _to_tensor_rgb01(img)


@torch.no_grad()
def _batch_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """pred, gt: [B,C,H,W] in [0,1] -> return [B] PSNR"""
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


class _EMAEvalCtx:
    """with _EMAEvalCtx(self.ema_helper, self.model): 切到 EMA 权重做验证，退出后自动恢复"""

    def __init__(self, ema_helper, model):
        self.ema = ema_helper
        self.model = model
        self.enabled = ema_helper is not None
        self.mode = None

    def __enter__(self):
        if not self.enabled:
            return self
        # 兼容两类 EMA 接口
        if hasattr(self.ema, "store") and hasattr(self.ema, "copy_to"):
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            self.mode = "params"
        elif hasattr(self.ema, "apply_shadow"):
            self.ema.apply_shadow(self.model)
            self.mode = "shadow"
        else:
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return
        if self.mode == "params":
            try:
                self.ema.restore(self.model.parameters())
            except TypeError:
                self.ema.restore(self.model)
        elif self.mode == "shadow":
            self.ema.restore(self.model)


# =========================
# EMA Helper
# =========================
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# =========================
# beta 调度
# =========================
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                             num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
                                  num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# =========================
# 主网络
# =========================
class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)

        # 训练时从 stage1 初始化 decom；评估时权重从 ddm ckpt 里加载
        if getattr(self.args, "mode", "training") == 'training':
            self.decom = self.load_stage1(CTDN(), 'ckpt/stage1')
        else:
            self.decom = CTDN()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        # ============ PGFM + CEG 模块 ============
        base_ch = getattr(self.config.model, 'ch', 64)
        self.enc_x = nn.Sequential(nn.Conv2d(3, base_ch, 3, 1, 1), nn.GELU())
        self.dec_x = nn.Sequential(nn.Conv2d(base_ch, 3, 3, 1, 1))
        self.pgfm = PGFM(ch=base_ch).to(self.device)
        self.ceg = CEG(ch=base_ch, n_expert=4, topk=2).to(self.device)

        # evaluation 模式下用于给中间图命名的计数器
        self.debug_counter = 0

    @staticmethod
    def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        beta: [T]
        t   : [B] ，包含时间步索引
        返回 alpha_bar_t，形状 [B,1,1,1]
        """
        if not torch.is_tensor(beta):
            beta = torch.as_tensor(beta, dtype=torch.float32)
        beta = beta.to(t.device)
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        """
        尝试优先加载绝对路径；否则从 model_dir 拼 'stage1_weight.pth.tar'
        """
        abs_path = "/home/ubuntu/project1/LightenDiffusion-main/ckpt/stage1/stage1_weight.pth.tar"
        cand_path = abs_path if os.path.isfile(abs_path) else os.path.join(
            model_dir, "stage1_weight.pth.tar")
        checkpoint = utils.logging.load_checkpoint(cand_path, 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    # -------------------------------------
    # 采样（训练 / 推理共用）：中间插 PGFM + CEG
    # -------------------------------------
    def sample_training(
        self,
        x_cond: torch.Tensor,
        eta: float = 0.0,
        low_img: torch.Tensor = None,
        deterministic: bool = False,
        noise_seed: int = 12345,
    ):
        """
        x_cond: 条件特征 [B,C,H,W] （通常是 low_condition_norm）
        low_img: 原始低光 RGB 图像 [B,3,H0,W0] ，用于 CTDN 分解提供先验
        返回:
          - 最终特征 pred_fea  (与 x_cond 分辨率一致)
          - aux_accum: 聚合的正则项（张量）
        """
        assert low_img is not None, "sample_training() 调用时请传 low_img=inputs"

        # 在函数内部统一构造 betas
        b = self.betas.to(x_cond.device)

        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = list(range(0, self.config.diffusion.num_diffusion_timesteps, skip))
        n, c, h, w = x_cond.shape
        seq_next = [-1] + seq[:-1]

        # —— 根据 deterministic 生成噪声 ——
        if deterministic:
            g = torch.Generator(device=self.device)
            g.manual_seed(noise_seed)
            x = torch.randn(n, c, h, w, device=self.device, generator=g)
        else:
            x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]

        # 每隔多少个时间步启用一次“CTDN + PGFM + CEG”
        SKIP_FACTOR = 4

        # 正则累计（张量，便于 DDP 聚合）
        aux_accum = torch.zeros(1, device=x_cond.device)

        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            # i, j 都是 int（扩散时间步索引）
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            next_t = torch.full((n,), j, device=x.device, dtype=torch.long)

            at = self.compute_alpha(b, t)
            at_next = self.compute_alpha(b, next_t)
            xt = xs[-1].to(x.device)

            # 常规反推：预测噪声 et，得到当前步的 x0_t（[-1,1]）
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # 先做一个粗筛：只有 i 能被 4 整除时才考虑用中间模块
            use_mid_gate = (i % 4 == 0)
            if not use_mid_gate:
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to(x.device))
                continue

            # 再用 SKIP_FACTOR 控制频率
            use_mid = (step_idx % SKIP_FACTOR == 0)
            if use_mid:
                # [-1,1] -> [0,1]
                x0_01 = torch.clamp((x0_t + 1) / 2.0, 0.0, 1.0)

                # 低光 3 通道 & [0,1]
                low_rgb = low_img[:, :3, ...] if low_img.shape[1] >= 3 else low_img
                low_01 = torch.clamp(low_rgb, 0.0, 1.0)

                # 对齐到 x0 分辨率
                Hf, Wf = x0_01.shape[-2], x0_01.shape[-1]
                if low_01.shape[-2:] != (Hf, Wf):
                    low_01 = F.interpolate(
                        low_01, size=(Hf, Wf), mode='bilinear', align_corners=False)

                pair = torch.cat([low_01, x0_01], dim=1)  # [B,6,Hf,Wf]

                # 教师分解（CTDN / RICD）
                with torch.no_grad():
                    out_ctdn = self.decom(pair, pred_fea=None)
                    teacher_R = out_ctdn["high_R"]
                    teacher_L = out_ctdn["high_L"]

                # L 单通道 → 3 通道；teacher 上采样到 (Hf, Wf)
                if teacher_L.shape[1] == 1:
                    teacher_L = teacher_L.repeat(1, 3, 1, 1)
                if teacher_R.shape[-2:] != (Hf, Wf):
                    teacher_R = F.interpolate(
                        teacher_R, size=(Hf, Wf), mode='bilinear', align_corners=False)
                if teacher_L.shape[-2:] != (Hf, Wf):
                    teacher_L = F.interpolate(
                        teacher_L, size=(Hf, Wf), mode='bilinear', align_corners=False)
                teacher_R = teacher_R.to(x0_01.device)
                teacher_L = teacher_L.to(x0_01.device)

                # ====== PGFM + CEG ======
                feat_in = self.enc_x(x0_01)

                # 先经过 PGFM
                feat_pgfm, loss_pgfm, _ = self.pgfm(
                    feat_in, teacher_R, teacher_L)

                # 再经过 CEG
                feat_ceg, loss_ceg = self.ceg(feat_pgfm, t_scalar=t.float())

                # CEG 后结果映射回 [-1,1]
                x0_ceg_01 = torch.clamp(self.dec_x(feat_ceg), 0.0, 1.0)
                x0_refined = x0_ceg_01 * 2.0 - 1.0

                # ====== 累计附加正则（兼容新旧 key） ======
                # CEG keys: ceg_balance/ceg_sparsity (new) OR moe_balance/moe_sparsity (old)
                ceg_bal = loss_ceg.get("ceg_balance", loss_ceg.get("moe_balance", 0.0))
                ceg_sp = loss_ceg.get("ceg_sparsity", loss_ceg.get("moe_sparsity", 0.0))
                # PGFM keys: pgfm_R/pgfm_L (new) OR rsam_R/rsam_L (old)
                pgfm_r = loss_pgfm.get("pgfm_R", loss_pgfm.get("rsam_R", 0.0))
                pgfm_l = loss_pgfm.get("pgfm_L", loss_pgfm.get("rsam_L", 0.0))

                aux_accum = aux_accum + (
                    ceg_bal + 5e-4 * ceg_sp
                    + 0.3 * pgfm_r + 0.1 * pgfm_l
                ).reshape(1)

                x0_used = x0_refined
            else:
                x0_used = x0_t

            # 合成下一步 x_{t-1}
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_used + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1], aux_accum

    # -------------------------------------
    # 可视化：给一整个 batch 生成 PGFM / CEG 中间图
    # -------------------------------------
    def dump_pgfm_ceg_batch(self, low_batch, clean_batch, save_root, names):
        """
        给定一个 batch 的低光图 low_batch 和增强结果 clean_batch（都在 [0,1]，[B,3,H,W]），
        为 batch 里的每一张图片分别生成：
          - debug_pgfm/<name>  ：只经过 PGFM 后的图
          - debug_ceg/<name>   ：PGFM + CEG 后的图
        """
        from torchvision.utils import save_image

        os.makedirs(save_root, exist_ok=True)
        pgfm_dir = os.path.join(save_root, "debug_pgfm")
        ceg_dir = os.path.join(save_root, "debug_ceg")
        os.makedirs(pgfm_dir, exist_ok=True)
        os.makedirs(ceg_dir, exist_ok=True)

        device = clean_batch.device
        B = clean_batch.size(0)

        # 统一到 [0,1]，3 通道 & 相同分辨率
        low_01 = low_batch[:, :3, ...].clamp(0.0, 1.0)
        x0_01 = clean_batch[:, :3, ...].clamp(0.0, 1.0)

        Hf, Wf = x0_01.shape[-2], x0_01.shape[-1]
        if low_01.shape[-2:] != (Hf, Wf):
            low_01 = F.interpolate(low_01, size=(Hf, Wf),
                                   mode='bilinear', align_corners=False)

        # CTDN / RICD 教师分解，输入为 [low, clean]
        pair = torch.cat([low_01, x0_01], dim=1)  # [B,6,H,W]
        with torch.no_grad():
            out_ctdn = self.decom(pair, pred_fea=None)
            teacher_R = out_ctdn["high_R"]
            teacher_L = out_ctdn["high_L"]

        # L 单通道 → 3 通道；teacher 上采样到 (Hf, Wf)
        if teacher_L.shape[1] == 1:
            teacher_L = teacher_L.repeat(1, 3, 1, 1)
        if teacher_R.shape[-2:] != (Hf, Wf):
            teacher_R = F.interpolate(
                teacher_R, size=(Hf, Wf), mode='bilinear', align_corners=False)
        if teacher_L.shape[-2:] != (Hf, Wf):
            teacher_L = F.interpolate(
                teacher_L, size=(Hf, Wf), mode='bilinear', align_corners=False)
        teacher_R = teacher_R.to(device)
        teacher_L = teacher_L.to(device)

        # ====== 先走一遍 PGFM ======
        feat_in = self.enc_x(x0_01)
        feat_pgfm, _, _ = self.pgfm(feat_in, teacher_R, teacher_L)

        # ====== 在 PGFM 基础上再走 CEG ======
        t_scalar = torch.zeros(B, device=device)  # 给门控一个时间标量
        feat_ceg, _ = self.ceg(feat_pgfm, t_scalar=t_scalar)

        # 解码并做 min-max 归一化到 [0,1]，让图更亮更易看
        x0_pgfm = self.dec_x(feat_pgfm)
        x0_ceg = self.dec_x(feat_ceg)

        mn_pgfm = x0_pgfm.amin(dim=(1, 2, 3), keepdim=True)
        mx_pgfm = x0_pgfm.amax(dim=(1, 2, 3), keepdim=True)
        x0_pgfm_01 = (x0_pgfm - mn_pgfm) / (mx_pgfm - mn_pgfm + 1e-8)

        mn_ceg = x0_ceg.amin(dim=(1, 2, 3), keepdim=True)
        mx_ceg = x0_ceg.amax(dim=(1, 2, 3), keepdim=True)
        x0_ceg_01 = (x0_ceg - mn_ceg) / (mx_ceg - mn_ceg + 1e-8)

        # 逐张保存，对应各自的文件名
        for b in range(B):
            name = names[b]
            pgfm_path = os.path.join(pgfm_dir, name)
            ceg_path = os.path.join(ceg_dir, name)
            save_image(x0_pgfm_01[b:b + 1].detach().cpu(), pgfm_path)
            save_image(x0_ceg_01[b:b + 1].detach().cpu(), ceg_path)

        print(f"[DEBUG] dumped PGFM/CEG intermediates for batch to {pgfm_dir} / {ceg_dir}")

    # -------------------------------------
    # 可视化：进入扩散模型之前的 R×L
    # -------------------------------------
    def dump_rl_before_batch(self, rl_batch, save_root, names):
        """
        rl_batch: [B, C, H, W] in [0,1]，通常是 low_R * high_L
        """
        from torchvision.utils import save_image

        os.makedirs(save_root, exist_ok=True)
        rl_dir = os.path.join(save_root, "debug_rl_before")
        os.makedirs(rl_dir, exist_ok=True)

        rl = rl_batch
        if rl.ndim != 4:
            return
        if rl.size(1) == 1:
            rl = rl.repeat(1, 3, 1, 1)

        B = rl.size(0)
        for b in range(B):
            img = rl[b:b + 1].clamp(0.0, 1.0)
            mn = img.amin(dim=(1, 2, 3), keepdim=True)
            mx = img.amax(dim=(1, 2, 3), keepdim=True)
            img = (img - mn) / (mx - mn + 1e-8)
            path = os.path.join(rl_dir, names[b])
            save_image(img.detach().cpu(), path)

        print(f"[DEBUG] dumped RL-before (R×L) maps to {rl_dir}")

    # -------------------------------------
    # 可视化：扩散去噪后、解码器之前的 pred_fea
    # -------------------------------------
    def dump_predfea_before_dec_batch(self, fea_batch, save_root, names):
        """
        fea_batch: [B, C, H, W]，通常是 inverse_data_transform 之后的 pred_fea
        """
        from torchvision.utils import save_image

        os.makedirs(save_root, exist_ok=True)
        pf_dir = os.path.join(save_root, "debug_predfea_before")
        os.makedirs(pf_dir, exist_ok=True)

        fea = fea_batch
        if fea.ndim != 4:
            return

        B, C, H, W = fea.shape
        # 如果不是 3 通道，简单处理一下，可视化前 3 通道或复制单通道
        if C == 1:
            fea_vis = fea.repeat(1, 3, 1, 1)
        elif C >= 3:
            fea_vis = fea[:, :3, :, :]
        else:
            # 理论上不会出现 C=0 等情况
            fea_vis = fea.repeat(1, 3, 1, 1)

        for b in range(B):
            img = fea_vis[b:b + 1]
            # 做一次 per-image 的 min-max 归一化，提升可视对比度
            mn = img.amin(dim=(1, 2, 3), keepdim=True)
            mx = img.amax(dim=(1, 2, 3), keepdim=True)
            img = (img - mn) / (mx - mn + 1e-8)
            path = os.path.join(pf_dir, names[b])
            save_image(img.detach().cpu(), path)

        print(f"[DEBUG] dumped pred_fea-before-dec maps to {pf_dir}")

    # -------------------------------------
    # forward
    # -------------------------------------
    def forward(self, inputs):
        data_dict = {}

        b = self.betas.to(inputs.device)

        if self.training:
            # ======== 训练分支 ========
            output = self.decom(inputs, pred_fea=None)
            low_R = output["low_R"]
            low_L = output["low_L"]
            low_fea = output["low_fea"]
            high_L = output["high_L"]

            low_condition_norm = utils.data_transform(low_fea)

            t = torch.randint(
                low=0,
                high=self.num_timesteps,
                size=(low_condition_norm.shape[0] // 2 + 1,),
                device=self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[
                :low_condition_norm.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(low_condition_norm)

            high_input_norm = utils.data_transform(low_R * high_L)
            x = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()

            noise_output = self.Unet(torch.cat([low_condition_norm, x], dim=1), t.float())

            # 解包 (pred, aux)
            pred_fea, aux_accum = self.sample_training(
                low_condition_norm, low_img=inputs, deterministic=False
            )
            pred_fea = utils.inverse_data_transform(pred_fea)
            reference_fea = low_R * torch.pow(low_L, 0.2)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pred_fea"] = pred_fea
            data_dict["reference_fea"] = reference_fea
            data_dict["aux_loss"] = aux_accum

        else:
            # ======== 推理 / 验证分支 ========
            output = self.decom(inputs, pred_fea=None)
            low_fea = output["low_fea"]
            low_condition_norm = utils.data_transform(low_fea)

            # 进入扩散模型之前的 R×L（用于可视化）
            low_R = output.get("low_R", None)
            high_L = output.get("high_L", None)
            rl_before = None
            if low_R is not None and high_L is not None:
                rl_before = (low_R * high_L).clamp(0.0, 1.0)

            pred_fea, _ = self.sample_training(
                low_condition_norm,
                low_img=inputs,
                deterministic=True,
                noise_seed=0,
            )

            pred_fea = utils.inverse_data_transform(pred_fea)
            # 先把 pred_fea 放进 data_dict，方便外面用
            data_dict["pred_fea"] = pred_fea

            pred_x = self.decom(inputs, pred_fea=pred_fea)["pred_img"]  # [B,3,H,W] in [0,1]
            data_dict["pred_x"] = pred_x
            if rl_before is not None:
                data_dict["rl_before"] = rl_before

            # -------- 非 training 模式下：直接在这里保存中间图 --------
            try:
                # 只要不是 training 模式（如 evaluation / restoration），就保存中间可视化
                if getattr(self.args, "mode", "") != "training":
                    save_root = getattr(self.args, "image_folder", "./results_debug")
                    B = pred_x.size(0)
                    names = []
                    for _ in range(B):
                        name = f"{self.debug_counter:05d}.png"
                        names.append(name)
                        self.debug_counter += 1

                    # 1) R×L 可视化
                    if rl_before is not None:
                        self.dump_rl_before_batch(rl_before.detach(), save_root, names)

                    # 2) 扩散去噪后、解码前的 pred_fea 可视化
                    self.dump_predfea_before_dec_batch(pred_fea.detach(), save_root, names)

                    # 3) PGFM / CEG 中间图
                    self.dump_pgfm_ceg_batch(inputs, pred_x.detach(), save_root, names)
            except Exception as e:
                # 出问题也不要影响推理
                print(f"[WARN] dump intermediates failed in forward: {e}")

        return data_dict


# =========================
# 训练封装
# =========================
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model = torch.nn.DataParallel(
            self.model, device_ids=range(torch.cuda.device_count()))
        self.model.to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(
            self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        if 'epoch' in checkpoint:
            self.start_epoch = int(checkpoint['epoch'])
        if 'step' in checkpoint:
            self.step = int(checkpoint['step'])

        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'ema_helper' in checkpoint and checkpoint['ema_helper'] is not None:
            try:
                self.ema_helper.load_state_dict(checkpoint['ema_helper'])
                if ema:
                    self.ema_helper.ema(self.model)
            except Exception:
                pass
        elif ema:
            self.ema_helper.ema(self.model)

        print("=> loaded checkpoint {} (epoch {}, step {})".format(
            load_path, getattr(self, 'start_epoch', 0), getattr(self, 'step', 0)))

    def train(self, DATASET):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        # 断点续训
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        # 冻结 decom（CTDN）
        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # ===== ETA 追踪器 =====
        steps_per_epoch = len(train_loader)
        total_steps = self.config.training.n_epochs * steps_per_epoch
        if not hasattr(self, "_eta_buf"):
            self._eta_buf = deque(maxlen=200)     # 最近 200 step 的滑动平均
            self._val_time_buf = deque(maxlen=8)  # 最近几次验证+保存耗时

        # best / early-stop / plateau（只初始化一次）
        if not hasattr(self, "_best_psnr"):
            self._best_psnr = -1.0
            self._early_best = -1e9
            self._bad_cnt = 0
            self._early_patience = 8
            self._plateau = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                              patience=3, min_lr=1e-6)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                # step 计时开始（含 GPU 同步）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start = time.time()

                x = x.to(self.device)
                output = self.model(x)  # dict

                noise_loss, scc_loss = self.noise_estimation_loss(output)
                loss = noise_loss + scc_loss

                # 辅助损失：在 60% 训练进度后线性拉起到 0.03
                if "aux_loss" in output and output["aux_loss"] is not None:
                    ratio = self.step / float(total_steps + 1e-8)
                    if ratio < 0.6:
                        W = 0.0
                    else:
                        W = 0.03 * min(1.0, (ratio - 0.6) / 0.4)
                    loss = loss + W * output["aux_loss"].mean()

                data_time += time.time() - data_start

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # EMA
                if hasattr(self, "ema_helper"):
                    self.ema_helper.update(self.model)
                elif hasattr(self, "ema"):
                    self.ema.update(self.model)

                # step 计时结束（含 GPU 同步）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_cost = time.time() - step_start
                self._eta_buf.append(step_cost)

                # === 每 10 step 打印一次 ETA ===
                if self.step % 10 == 0:
                    avg_step = (sum(self._eta_buf) /
                                len(self._eta_buf)) if len(self._eta_buf) > 0 else step_cost
                    remain_steps = max(0, total_steps - self.step)
                    # 预估剩余会发生几次验证
                    vf = self.config.training.validation_freq
                    remain_validations = math.floor(
                        remain_steps / max(1, vf))
                    avg_val = (sum(self._val_time_buf) /
                               len(self._val_time_buf)) if len(self._val_time_buf) > 0 else 0.0
                    eta_seconds = int(
                        avg_step * remain_steps + avg_val * remain_validations)
                    td = datetime.timedelta(seconds=eta_seconds)
                    days = td.days
                    h = td.seconds // 3600
                    m = (td.seconds % 3600) // 60
                    s = td.seconds % 60
                    eta_str = f"{days}d {h}h {m}m {s}s" if days > 0 else f"{h}h {m}m {s}s"

                    print("step:{}/{} | noise_loss:{:.5f} | scc_loss:{:.5f} | ETA:{}"
                          .format(self.step, total_steps, noise_loss.item(), scc_loss.item(), eta_str))

                data_start = time.time()

                # —— 验证 / 保存 ——
                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _val_start = time.time()

                    # 1) 用 EMA 权重做验证，并抓取一份 EMA 的 state_dict
                    with _EMAEvalCtx(getattr(self, "ema_helper", None) or getattr(self, "ema", None), self.model):
                        self.model.eval()
                        psnr_val = self.sample_validation_patches(
                            val_loader, self.step)
                        # 抓取 EMA 权重（with 内参数即为 EMA）
                        ema_state_dict = {k: v.detach().cpu().clone()
                                          for k, v in self.model.state_dict().items()}
                        self.model.train()

                    # 2) 每次验证都保存“当步快照”（非 EMA）
                    _save_ckpt_exact({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),   # 注意：此时已恢复到非 EMA
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                        'params': self.args,
                        'config': self.config
                    }, os.path.join(self.config.data.ckpt_dir, f'model_step_{self.step}.pth.tar'))

                    # 3) 如果 PSNR 提升，则刷新 best（存两份：非EMA + EMA）
                    if not hasattr(self, "_best_psnr"):
                        self._best_psnr = -1.0
                    print(
                        f"[VAL] step {self.step}  PSNR={psnr_val:.4f}  (best={self._best_psnr:.4f})")

                    if psnr_val > self._best_psnr + 1e-4:
                        self._best_psnr = psnr_val

                        # 3.1 最佳（非 EMA）
                        _save_ckpt_exact({
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),   # 非 EMA
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                            'params': self.args,
                            'config': self.config
                        }, os.path.join(self.config.data.ckpt_dir, 'model_best_val.pth.tar'))

                        # 3.2 最佳（EMA）—— 建议评测用这个
                        _save_ckpt_exact({
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': ema_state_dict,  # 直接存 EMA 权重
                            'optimizer': None,
                            'ema_helper': None,
                            'params': self.args,
                            'config': self.config
                        }, os.path.join(self.config.data.ckpt_dir, 'model_best_val_ema.pth.tar'))

                        self._bad_cnt = 0
                        print(
                            f"[BEST UPDATED] step {self.step}  new best PSNR={self._best_psnr:.4f}")
                    else:
                        self._bad_cnt += 1

                    # 4) 每次验证都滚动保存 latest（非 EMA，便于断点续训）
                    _save_ckpt_exact({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),   # 非 EMA
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                        'params': self.args,
                        'config': self.config
                    }, os.path.join(self.config.data.ckpt_dir, 'model_latest.pth.tar'))

                    # 记录验证耗时（ETA 用）
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _val_cost = time.time() - _val_start
                    self._val_time_buf.append(_val_cost)

    def noise_estimation_loss(self, output):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]
        noise_loss = self.l2_loss(noise_output, e)
        scc_loss = 0.001 * self.l1_loss(pred_fea, reference_fea)
        return noise_loss, scc_loss

    @torch.no_grad()
    def sample_validation_patches(self, val_loader, step):
        """
        生成验证集结果到 <image_folder>/<step>/ 下，并返回平均 PSNR（float）
        - 对每一张验证图，额外生成：
            debug_pgfm/<name>  和  debug_ceg/<name>
        """
        self.model.eval()
        device = self.device

        save_root = os.path.join(self.args.image_folder, str(step))
        os.makedirs(save_root, exist_ok=True)

        # 推断 GT 根目录：优先 val_dir/high；否则 val_dir
        gt_root = None
        if hasattr(self.config, "data") and hasattr(self.config.data, "val_dir"):
            cand = os.path.join(self.config.data.val_dir, "high")
            gt_root = cand if os.path.isdir(cand) else (
                self.config.data.val_dir if os.path.isdir(self.config.data.val_dir) else None
            )

        psnr_list = []

        # 取到底层的 Net（DataParallel 包了一层）
        net = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model

        for bi, (x, y) in enumerate(val_loader):
            B, C, H, W = x.shape
            # pad 到 64 的倍数
            H64 = int(64 * np.ceil(H / 64.0))
            W64 = int(64 * np.ceil(W / 64.0))
            x_pad = F.pad(x, (0, W64 - W, 0, H64 - H),
                          mode='reflect').to(device, non_blocking=True)

            # 前向推理
            out = self.model(x_pad)
            if isinstance(out, dict) and "pred_x" in out:
                pred = out["pred_x"]
            else:
                pred = out
            # 裁回原图尺寸并夹紧
            pred = torch.clamp(pred[..., :H, :W], 0.0, 1.0)  # [B,3,H,W]

            # GT
            if torch.is_tensor(y):
                gt = y.to(device, non_blocking=True)
            else:
                assert gt_root is not None, "无法定位 GT 根目录（请检查 config.data.val_dir）"
                gts = []
                for j in range(len(y)):
                    gt_j = _load_gt_tensor(y[j], gt_root)  # [3,H,W]
                    gts.append(gt_j)
                gt = torch.stack(gts, 0).to(device)

            # 尺寸对齐（裁到共同最小 H,W）
            HH = min(pred.size(-2), gt.size(-2))
            WW = min(pred.size(-1), gt.size(-1))
            pred_c = pred[..., :HH, :WW]
            gt_c = gt[..., :HH, :WW]

            # 计算 PSNR
            ps = _batch_psnr(pred_c, gt_c)  # [B]
            psnr_list.append(ps.detach().cpu())

            # ==== 保存增强结果，同时构造每张图的文件名 ====
            batch_names = []
            for b in range(B):
                if isinstance(y, (list, tuple)):
                    name = os.path.basename(str(y[b]))
                else:
                    name = f"{bi:05d}_{b:02d}.png"
                batch_names.append(name)
                torchvision.utils.save_image(
                    pred[b].detach().cpu(),
                    os.path.join(save_root, name)
                )

            # ==== 对这个 batch 额外生成 PGFM / CEG 中间图 ====
            if hasattr(net, "dump_pgfm_ceg_batch"):
                net.dump_pgfm_ceg_batch(
                    x_pad.to(device),       # 低光输入（pad 后）
                    pred.to(device),        # 增强输出
                    save_root,              # 当前 step 子目录
                    batch_names             # 文件名对齐
                )

        psnr_avg = torch.cat(psnr_list, 0).mean().item() if len(psnr_list) else 0.0
        self.model.train()
        return psnr_avg
