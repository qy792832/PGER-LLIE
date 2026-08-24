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
from models.decom import RICD, ReconstructionDecoder
from models.pgfm import PGFM
from models.ceg import CEG


def _save_ckpt_exact(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)  # ä¸åšä»»ä½•äºŒæ¬¡å‘½å


# =========================
# å·¥å…·ï¼šå›¾åƒ/PSNR/EMA éªŒè¯ä¸Šä¸‹æ–‡
# =========================
def _to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    """PIL -> Tensor [C,H,W], 0~1"""
    arr = np.array(img.convert('RGB'), dtype=np.uint8)  # H,W,3
    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return ten


@torch.no_grad()
def _load_gt_tensor(y_item: Union[str, os.PathLike], gt_root: str) -> torch.Tensor:
    """y_item å¯èƒ½æ˜¯æ–‡ä»¶åæˆ–ç›¸å¯¹è·¯å¾„ï¼›gt_root é€šå¸¸æ˜¯ val_dir/high æˆ– val_dirã€‚"""
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
    """with _EMAEvalCtx(self.ema_helper, self.model): åˆ‡åˆ° EMA æƒé‡åšéªŒè¯ï¼Œé€€å‡ºåŽè‡ªåŠ¨æ¢å¤"""

    def __init__(self, ema_helper, model):
        self.ema = ema_helper
        self.model = model
        self.enabled = ema_helper is not None
        self.mode = None

    def __enter__(self):
        if not self.enabled:
            return self
        # å…¼å®¹ä¸¤ç±» EMA æŽ¥å£
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
# beta è°ƒåº¦
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
# ä¸»ç½‘ç»œ
# =========================
class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)

        # Streamlined RICD teacher and separated reconstruction decoder
        self.decom = RICD(channels=64)
        self.recon_decoder = ReconstructionDecoder(channels=64)

        # During Stage-II training, initialize both components from Stage-I CTDN
        if getattr(self.args, "mode", "training") == "training":
            self.load_stage1(
                self.decom,
                self.recon_decoder,
                "ckpt/stage1",
            )

        # Both Stage-I components remain frozen during Stage II
        for module in (self.decom, self.recon_decoder):
            for param in module.parameters():
                param.requires_grad = False

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        # ============ PGFM + CEG æ¨¡å— ============
        base_ch = getattr(self.config.model, 'ch', 64)
        self.enc_x = nn.Sequential(nn.Conv2d(3, base_ch, 3, 1, 1), nn.GELU())
        self.dec_x = nn.Sequential(nn.Conv2d(base_ch, 3, 3, 1, 1))
        self.pgfm = PGFM(ch=base_ch).to(self.device)
        self.ceg = CEG(ch=base_ch, n_expert=4, topk=2).to(self.device)

        # evaluation æ¨¡å¼ä¸‹ç”¨äºŽç»™ä¸­é—´å›¾å‘½åçš„è®¡æ•°å™¨
        self.debug_counter = 0

    @staticmethod
    def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        beta: [T]
        t   : [B] ï¼ŒåŒ…å«æ—¶é—´æ­¥ç´¢å¼•
        è¿”å›ž alpha_bar_tï¼Œå½¢çŠ¶ [B,1,1,1]
        """
        if not torch.is_tensor(beta):
            beta = torch.as_tensor(beta, dtype=torch.float32)
        beta = beta.to(t.device)
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(ricd, recon_decoder, model_dir):
        """
        Load a full Stage-I CTDN checkpoint into the streamlined RICD
        and the separated reconstruction decoder.
        """
        checkpoint_path = os.path.join(
            model_dir,
            "stage1_weight.pth.tar",
        )
        checkpoint = utils.logging.load_checkpoint(
            checkpoint_path,
            "cuda",
        )

        state_dict = checkpoint.get(
            "model",
            checkpoint.get("state_dict", checkpoint),
        )

        normalized_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            normalized_state[key] = value

        ricd_state = {}
        decoder_state = {}

        for key, value in normalized_state.items():
            if (
                key.startswith("ReconNet.pyramid.")
                or key.startswith("ReconNet.channel_down.")
                or key.startswith("retinex.")
            ):
                ricd_state[key] = value
            elif key.startswith("ReconNet."):
                decoder_key = key[len("ReconNet."):]
                decoder_state[decoder_key] = value

        ricd.load_state_dict(ricd_state, strict=True)
        recon_decoder.load_state_dict(decoder_state, strict=True)

        print(
            "=> loaded Stage-I CTDN weights into streamlined "
            "RICD and reconstruction decoder"
        )


    # -------------------------------------
    # é‡‡æ ·ï¼ˆè®­ç»ƒ / æŽ¨ç†å…±ç”¨ï¼‰ï¼šä¸­é—´æ’ PGFM + CEG
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
        x_cond: æ¡ä»¶ç‰¹å¾ [B,C,H,W] ï¼ˆé€šå¸¸æ˜¯ low_condition_normï¼‰
        low_img: åŽŸå§‹ä½Žå…‰ RGB å›¾åƒ [B,3,H0,W0] ï¼Œç”¨äºŽ CTDN åˆ†è§£æä¾›å…ˆéªŒ
        è¿”å›ž:
          - æœ€ç»ˆç‰¹å¾ pred_fea  (ä¸Ž x_cond åˆ†è¾¨çŽ‡ä¸€è‡´)
          - aux_accum: èšåˆçš„æ­£åˆ™é¡¹ï¼ˆå¼ é‡ï¼‰
        """
        assert low_img is not None, "sample_training() è°ƒç”¨æ—¶è¯·ä¼  low_img=inputs"

        # åœ¨å‡½æ•°å†…éƒ¨ç»Ÿä¸€æž„é€  betas
        b = self.betas.to(x_cond.device)

        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = list(range(0, self.config.diffusion.num_diffusion_timesteps, skip))
        n, c, h, w = x_cond.shape
        seq_next = [-1] + seq[:-1]

        # â€”â€” æ ¹æ® deterministic ç”Ÿæˆå™ªå£° â€”â€”
        if deterministic:
            g = torch.Generator(device=self.device)
            g.manual_seed(noise_seed)
            x = torch.randn(n, c, h, w, device=self.device, generator=g)
        else:
            x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]

        # æ¯éš”å¤šå°‘ä¸ªæ—¶é—´æ­¥å¯ç”¨ä¸€æ¬¡â€œCTDN + PGFM + CEGâ€
        SKIP_FACTOR = 4

        # æ­£åˆ™ç´¯è®¡ï¼ˆå¼ é‡ï¼Œä¾¿äºŽ DDP èšåˆï¼‰
        aux_accum = torch.zeros(1, device=x_cond.device)

        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            # i, j éƒ½æ˜¯ intï¼ˆæ‰©æ•£æ—¶é—´æ­¥ç´¢å¼•ï¼‰
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            next_t = torch.full((n,), j, device=x.device, dtype=torch.long)

            at = self.compute_alpha(b, t)
            at_next = self.compute_alpha(b, next_t)
            xt = xs[-1].to(x.device)

            # å¸¸è§„åæŽ¨ï¼šé¢„æµ‹å™ªå£° etï¼Œå¾—åˆ°å½“å‰æ­¥çš„ x0_tï¼ˆ[-1,1]ï¼‰
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # å…ˆåšä¸€ä¸ªç²—ç­›ï¼šåªæœ‰ i èƒ½è¢« 4 æ•´é™¤æ—¶æ‰è€ƒè™‘ç”¨ä¸­é—´æ¨¡å—
            use_mid_gate = (i % 4 == 0)
            if not use_mid_gate:
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to(x.device))
                continue

            # å†ç”¨ SKIP_FACTOR æŽ§åˆ¶é¢‘çŽ‡
            use_mid = (step_idx % SKIP_FACTOR == 0)
            if use_mid:
                # [-1,1] -> [0,1]
                x0_01 = torch.clamp((x0_t + 1) / 2.0, 0.0, 1.0)

                # ä½Žå…‰ 3 é€šé“ & [0,1]
                low_rgb = low_img[:, :3, ...] if low_img.shape[1] >= 3 else low_img
                low_01 = torch.clamp(low_rgb, 0.0, 1.0)

                # å¯¹é½åˆ° x0 åˆ†è¾¨çŽ‡
                Hf, Wf = x0_01.shape[-2], x0_01.shape[-1]
                if low_01.shape[-2:] != (Hf, Wf):
                    low_01 = F.interpolate(
                        low_01, size=(Hf, Wf), mode='bilinear', align_corners=False)

                pair = torch.cat([low_01, x0_01], dim=1)  # [B,6,Hf,Wf]

                # æ•™å¸ˆåˆ†è§£ï¼ˆCTDN / RICDï¼‰
                with torch.no_grad():
                    out_ctdn = self.decom(pair, pred_fea=None)
                    teacher_R = out_ctdn["high_R"]
                    teacher_L = out_ctdn["high_L"]

                # L å•é€šé“ â†’ 3 é€šé“ï¼›teacher ä¸Šé‡‡æ ·åˆ° (Hf, Wf)
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

                # å…ˆç»è¿‡ PGFM
                feat_pgfm, loss_pgfm, _ = self.pgfm(
                    feat_in, teacher_R, teacher_L)

                # å†ç»è¿‡ CEG
                feat_ceg, loss_ceg = self.ceg(feat_pgfm, t_scalar=t.float())

                # CEG åŽç»“æžœæ˜ å°„å›ž [-1,1]
                x0_ceg_01 = torch.clamp(self.dec_x(feat_ceg), 0.0, 1.0)
                x0_refined = x0_ceg_01 * 2.0 - 1.0

                # ====== ç´¯è®¡é™„åŠ æ­£åˆ™ï¼ˆå…¼å®¹æ–°æ—§ keyï¼‰ ======
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

            # åˆæˆä¸‹ä¸€æ­¥ x_{t-1}
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_used + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1], aux_accum

    # -------------------------------------
    # å¯è§†åŒ–ï¼šç»™ä¸€æ•´ä¸ª batch ç”Ÿæˆ PGFM / CEG ä¸­é—´å›¾
    # -------------------------------------
    def dump_pgfm_ceg_batch(self, low_batch, clean_batch, save_root, names):
        """
        ç»™å®šä¸€ä¸ª batch çš„ä½Žå…‰å›¾ low_batch å’Œå¢žå¼ºç»“æžœ clean_batchï¼ˆéƒ½åœ¨ [0,1]ï¼Œ[B,3,H,W]ï¼‰ï¼Œ
        ä¸º batch é‡Œçš„æ¯ä¸€å¼ å›¾ç‰‡åˆ†åˆ«ç”Ÿæˆï¼š
          - debug_pgfm/<name>  ï¼šåªç»è¿‡ PGFM åŽçš„å›¾
          - debug_ceg/<name>   ï¼šPGFM + CEG åŽçš„å›¾
        """
        from torchvision.utils import save_image

        os.makedirs(save_root, exist_ok=True)
        pgfm_dir = os.path.join(save_root, "debug_pgfm")
        ceg_dir = os.path.join(save_root, "debug_ceg")
        os.makedirs(pgfm_dir, exist_ok=True)
        os.makedirs(ceg_dir, exist_ok=True)

        device = clean_batch.device
        B = clean_batch.size(0)

        # ç»Ÿä¸€åˆ° [0,1]ï¼Œ3 é€šé“ & ç›¸åŒåˆ†è¾¨çŽ‡
        low_01 = low_batch[:, :3, ...].clamp(0.0, 1.0)
        x0_01 = clean_batch[:, :3, ...].clamp(0.0, 1.0)

        Hf, Wf = x0_01.shape[-2], x0_01.shape[-1]
        if low_01.shape[-2:] != (Hf, Wf):
            low_01 = F.interpolate(low_01, size=(Hf, Wf),
                                   mode='bilinear', align_corners=False)

        # CTDN / RICD æ•™å¸ˆåˆ†è§£ï¼Œè¾“å…¥ä¸º [low, clean]
        pair = torch.cat([low_01, x0_01], dim=1)  # [B,6,H,W]
        with torch.no_grad():
            out_ctdn = self.decom(pair, pred_fea=None)
            teacher_R = out_ctdn["high_R"]
            teacher_L = out_ctdn["high_L"]

        # L å•é€šé“ â†’ 3 é€šé“ï¼›teacher ä¸Šé‡‡æ ·åˆ° (Hf, Wf)
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

        # ====== å…ˆèµ°ä¸€é PGFM ======
        feat_in = self.enc_x(x0_01)
        feat_pgfm, _, _ = self.pgfm(feat_in, teacher_R, teacher_L)

        # ====== åœ¨ PGFM åŸºç¡€ä¸Šå†èµ° CEG ======
        t_scalar = torch.zeros(B, device=device)  # ç»™é—¨æŽ§ä¸€ä¸ªæ—¶é—´æ ‡é‡
        feat_ceg, _ = self.ceg(feat_pgfm, t_scalar=t_scalar)

        # è§£ç å¹¶åš min-max å½’ä¸€åŒ–åˆ° [0,1]ï¼Œè®©å›¾æ›´äº®æ›´æ˜“çœ‹
        x0_pgfm = self.dec_x(feat_pgfm)
        x0_ceg = self.dec_x(feat_ceg)

        mn_pgfm = x0_pgfm.amin(dim=(1, 2, 3), keepdim=True)
        mx_pgfm = x0_pgfm.amax(dim=(1, 2, 3), keepdim=True)
        x0_pgfm_01 = (x0_pgfm - mn_pgfm) / (mx_pgfm - mn_pgfm + 1e-8)

        mn_ceg = x0_ceg.amin(dim=(1, 2, 3), keepdim=True)
        mx_ceg = x0_ceg.amax(dim=(1, 2, 3), keepdim=True)
        x0_ceg_01 = (x0_ceg - mn_ceg) / (mx_ceg - mn_ceg + 1e-8)

        # é€å¼ ä¿å­˜ï¼Œå¯¹åº”å„è‡ªçš„æ–‡ä»¶å
        for b in range(B):
            name = names[b]
            pgfm_path = os.path.join(pgfm_dir, name)
            ceg_path = os.path.join(ceg_dir, name)
            save_image(x0_pgfm_01[b:b + 1].detach().cpu(), pgfm_path)
            save_image(x0_ceg_01[b:b + 1].detach().cpu(), ceg_path)

        print(f"[DEBUG] dumped PGFM/CEG intermediates for batch to {pgfm_dir} / {ceg_dir}")

    # -------------------------------------
    # å¯è§†åŒ–ï¼šè¿›å…¥æ‰©æ•£æ¨¡åž‹ä¹‹å‰çš„ RÃ—L
    # -------------------------------------
    def dump_rl_before_batch(self, rl_batch, save_root, names):
        """
        rl_batch: [B, C, H, W] in [0,1]ï¼Œé€šå¸¸æ˜¯ low_R * high_L
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

        print(f"[DEBUG] dumped RL-before (RÃ—L) maps to {rl_dir}")

    # -------------------------------------
    # å¯è§†åŒ–ï¼šæ‰©æ•£åŽ»å™ªåŽã€è§£ç å™¨ä¹‹å‰çš„ pred_fea
    # -------------------------------------
    def dump_predfea_before_dec_batch(self, fea_batch, save_root, names):
        """
        fea_batch: [B, C, H, W]ï¼Œé€šå¸¸æ˜¯ inverse_data_transform ä¹‹åŽçš„ pred_fea
        """
        from torchvision.utils import save_image

        os.makedirs(save_root, exist_ok=True)
        pf_dir = os.path.join(save_root, "debug_predfea_before")
        os.makedirs(pf_dir, exist_ok=True)

        fea = fea_batch
        if fea.ndim != 4:
            return

        B, C, H, W = fea.shape
        # å¦‚æžœä¸æ˜¯ 3 é€šé“ï¼Œç®€å•å¤„ç†ä¸€ä¸‹ï¼Œå¯è§†åŒ–å‰ 3 é€šé“æˆ–å¤åˆ¶å•é€šé“
        if C == 1:
            fea_vis = fea.repeat(1, 3, 1, 1)
        elif C >= 3:
            fea_vis = fea[:, :3, :, :]
        else:
            # ç†è®ºä¸Šä¸ä¼šå‡ºçŽ° C=0 ç­‰æƒ…å†µ
            fea_vis = fea.repeat(1, 3, 1, 1)

        for b in range(B):
            img = fea_vis[b:b + 1]
            # åšä¸€æ¬¡ per-image çš„ min-max å½’ä¸€åŒ–ï¼Œæå‡å¯è§†å¯¹æ¯”åº¦
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
            # ======== è®­ç»ƒåˆ†æ”¯ ========
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

            # è§£åŒ… (pred, aux)
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
            # ======== æŽ¨ç† / éªŒè¯åˆ†æ”¯ ========
            output = self.decom(
                inputs,
                pred_fea=None,
                return_pyramid=True,
            )
            low_fea = output["low_fea"]
            low_condition_norm = utils.data_transform(low_fea)

            # è¿›å…¥æ‰©æ•£æ¨¡åž‹ä¹‹å‰çš„ RÃ—Lï¼ˆç”¨äºŽå¯è§†åŒ–ï¼‰
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
            # å…ˆæŠŠ pred_fea æ”¾è¿› data_dictï¼Œæ–¹ä¾¿å¤–é¢ç”¨
            data_dict["pred_fea"] = pred_fea

            pred_x = self.recon_decoder(
                pred_fea,
                output["low_pyramid"],
            )
            data_dict["pred_x"] = pred_x
            if rl_before is not None:
                data_dict["rl_before"] = rl_before

            # -------- éž training æ¨¡å¼ä¸‹ï¼šç›´æŽ¥åœ¨è¿™é‡Œä¿å­˜ä¸­é—´å›¾ --------
            try:
                # åªè¦ä¸æ˜¯ training æ¨¡å¼ï¼ˆå¦‚ evaluation / restorationï¼‰ï¼Œå°±ä¿å­˜ä¸­é—´å¯è§†åŒ–
                if getattr(self.args, "mode", "") != "training":
                    save_root = getattr(self.args, "image_folder", "./results_debug")
                    B = pred_x.size(0)
                    names = []
                    for _ in range(B):
                        name = f"{self.debug_counter:05d}.png"
                        names.append(name)
                        self.debug_counter += 1

                    # 1) RÃ—L å¯è§†åŒ–
                    if rl_before is not None:
                        self.dump_rl_before_batch(rl_before.detach(), save_root, names)

                    # 2) æ‰©æ•£åŽ»å™ªåŽã€è§£ç å‰çš„ pred_fea å¯è§†åŒ–
                    self.dump_predfea_before_dec_batch(pred_fea.detach(), save_root, names)

                    # 3) PGFM / CEG ä¸­é—´å›¾
                    self.dump_pgfm_ceg_batch(inputs, pred_x.detach(), save_root, names)
            except Exception as e:
                # å‡ºé—®é¢˜ä¹Ÿä¸è¦å½±å“æŽ¨ç†
                print(f"[WARN] dump intermediates failed in forward: {e}")

        return data_dict


# =========================
# è®­ç»ƒå°è£…
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

    @staticmethod
    def _remap_legacy_ctdn_keys(state_dict):
        """
        Move legacy CTDN reconstruction keys to the separated decoder.
        Retained RICD keys keep their original names.
        """
        remapped = {}
        migrated = False
        marker = "decom.ReconNet."
        retained_prefixes = (
            "pyramid.",
            "channel_down.",
        )

        for key, value in state_dict.items():
            new_key = key

            if marker in key:
                suffix = key.split(marker, 1)[1]
                if not suffix.startswith(retained_prefixes):
                    new_key = key.replace(
                        marker,
                        "recon_decoder.",
                        1,
                    )
                    migrated = True

            remapped[new_key] = value

        return remapped, migrated

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)

        state_dict, migrated = self._remap_legacy_ctdn_keys(
            checkpoint["state_dict"]
        )
        self.model.load_state_dict(state_dict, strict=True)

        if "epoch" in checkpoint:
            self.start_epoch = int(checkpoint["epoch"])
        if "step" in checkpoint:
            self.step = int(checkpoint["step"])

        # Legacy optimizer states use the old parameter ordering.
        if (
            not migrated
            and "optimizer" in checkpoint
            and checkpoint["optimizer"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        elif migrated:
            print(
                "[INFO] Legacy CTDN checkpoint migrated; "
                "optimizer state was not restored."
            )

        if (
            "ema_helper" in checkpoint
            and checkpoint["ema_helper"] is not None
        ):
            try:
                self.ema_helper.load_state_dict(
                    checkpoint["ema_helper"]
                )
                if ema:
                    self.ema_helper.ema(self.model)
            except Exception:
                pass
        elif ema:
            self.ema_helper.ema(self.model)

        print(
            "=> loaded checkpoint {} (epoch {}, step {})".format(
                load_path,
                getattr(self, "start_epoch", 0),
                getattr(self, "step", 0),
            )
        )

    def train(self, DATASET):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        # æ–­ç‚¹ç»­è®­
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        # Freeze the Stage-I RICD teacher and reconstruction decoder
        for name, param in self.model.named_parameters():
            if "decom" in name or "recon_decoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # ===== ETA è¿½è¸ªå™¨ =====
        steps_per_epoch = len(train_loader)
        total_steps = self.config.training.n_epochs * steps_per_epoch
        if not hasattr(self, "_eta_buf"):
            self._eta_buf = deque(maxlen=200)     # æœ€è¿‘ 200 step çš„æ»‘åŠ¨å¹³å‡
            self._val_time_buf = deque(maxlen=8)  # æœ€è¿‘å‡ æ¬¡éªŒè¯+ä¿å­˜è€—æ—¶

        # best / early-stop / plateauï¼ˆåªåˆå§‹åŒ–ä¸€æ¬¡ï¼‰
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

                # step è®¡æ—¶å¼€å§‹ï¼ˆå« GPU åŒæ­¥ï¼‰
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start = time.time()

                x = x.to(self.device)
                output = self.model(x)  # dict

                noise_loss, scc_loss = self.noise_estimation_loss(output)
                loss = noise_loss + scc_loss

                # è¾…åŠ©æŸå¤±ï¼šåœ¨ 60% è®­ç»ƒè¿›åº¦åŽçº¿æ€§æ‹‰èµ·åˆ° 0.03
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

                # step è®¡æ—¶ç»“æŸï¼ˆå« GPU åŒæ­¥ï¼‰
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_cost = time.time() - step_start
                self._eta_buf.append(step_cost)

                # === æ¯ 10 step æ‰“å°ä¸€æ¬¡ ETA ===
                if self.step % 10 == 0:
                    avg_step = (sum(self._eta_buf) /
                                len(self._eta_buf)) if len(self._eta_buf) > 0 else step_cost
                    remain_steps = max(0, total_steps - self.step)
                    # é¢„ä¼°å‰©ä½™ä¼šå‘ç”Ÿå‡ æ¬¡éªŒè¯
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

                # â€”â€” éªŒè¯ / ä¿å­˜ â€”â€”
                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _val_start = time.time()

                    # 1) ç”¨ EMA æƒé‡åšéªŒè¯ï¼Œå¹¶æŠ“å–ä¸€ä»½ EMA çš„ state_dict
                    with _EMAEvalCtx(getattr(self, "ema_helper", None) or getattr(self, "ema", None), self.model):
                        self.model.eval()
                        psnr_val = self.sample_validation_patches(
                            val_loader, self.step)
                        # æŠ“å– EMA æƒé‡ï¼ˆwith å†…å‚æ•°å³ä¸º EMAï¼‰
                        ema_state_dict = {k: v.detach().cpu().clone()
                                          for k, v in self.model.state_dict().items()}
                        self.model.train()

                    # 2) æ¯æ¬¡éªŒè¯éƒ½ä¿å­˜â€œå½“æ­¥å¿«ç…§â€ï¼ˆéž EMAï¼‰
                    _save_ckpt_exact({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),   # æ³¨æ„ï¼šæ­¤æ—¶å·²æ¢å¤åˆ°éž EMA
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                        'params': self.args,
                        'config': self.config
                    }, os.path.join(self.config.data.ckpt_dir, f'model_step_{self.step}.pth.tar'))

                    # 3) å¦‚æžœ PSNR æå‡ï¼Œåˆ™åˆ·æ–° bestï¼ˆå­˜ä¸¤ä»½ï¼šéžEMA + EMAï¼‰
                    if not hasattr(self, "_best_psnr"):
                        self._best_psnr = -1.0
                    print(
                        f"[VAL] step {self.step}  PSNR={psnr_val:.4f}  (best={self._best_psnr:.4f})")

                    if psnr_val > self._best_psnr + 1e-4:
                        self._best_psnr = psnr_val

                        # 3.1 æœ€ä½³ï¼ˆéž EMAï¼‰
                        _save_ckpt_exact({
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),   # éž EMA
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                            'params': self.args,
                            'config': self.config
                        }, os.path.join(self.config.data.ckpt_dir, 'model_best_val.pth.tar'))

                        # 3.2 æœ€ä½³ï¼ˆEMAï¼‰â€”â€” å»ºè®®è¯„æµ‹ç”¨è¿™ä¸ª
                        _save_ckpt_exact({
                            'step': self.step,
                            'epoch': epoch + 1,
                            'state_dict': ema_state_dict,  # ç›´æŽ¥å­˜ EMA æƒé‡
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

                    # 4) æ¯æ¬¡éªŒè¯éƒ½æ»šåŠ¨ä¿å­˜ latestï¼ˆéž EMAï¼Œä¾¿äºŽæ–­ç‚¹ç»­è®­ï¼‰
                    _save_ckpt_exact({
                        'step': self.step,
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),   # éž EMA
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': (self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None),
                        'params': self.args,
                        'config': self.config
                    }, os.path.join(self.config.data.ckpt_dir, 'model_latest.pth.tar'))

                    # è®°å½•éªŒè¯è€—æ—¶ï¼ˆETA ç”¨ï¼‰
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
        ç”ŸæˆéªŒè¯é›†ç»“æžœåˆ° <image_folder>/<step>/ ä¸‹ï¼Œå¹¶è¿”å›žå¹³å‡ PSNRï¼ˆfloatï¼‰
        - å¯¹æ¯ä¸€å¼ éªŒè¯å›¾ï¼Œé¢å¤–ç”Ÿæˆï¼š
            debug_pgfm/<name>  å’Œ  debug_ceg/<name>
        """
        self.model.eval()
        device = self.device

        save_root = os.path.join(self.args.image_folder, str(step))
        os.makedirs(save_root, exist_ok=True)

        # æŽ¨æ–­ GT æ ¹ç›®å½•ï¼šä¼˜å…ˆ val_dir/highï¼›å¦åˆ™ val_dir
        gt_root = None
        if hasattr(self.config, "data") and hasattr(self.config.data, "val_dir"):
            cand = os.path.join(self.config.data.val_dir, "high")
            gt_root = cand if os.path.isdir(cand) else (
                self.config.data.val_dir if os.path.isdir(self.config.data.val_dir) else None
            )

        psnr_list = []

        # å–åˆ°åº•å±‚çš„ Netï¼ˆDataParallel åŒ…äº†ä¸€å±‚ï¼‰
        net = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model

        for bi, (x, y) in enumerate(val_loader):
            B, C, H, W = x.shape
            # pad åˆ° 64 çš„å€æ•°
            H64 = int(64 * np.ceil(H / 64.0))
            W64 = int(64 * np.ceil(W / 64.0))
            x_pad = F.pad(x, (0, W64 - W, 0, H64 - H),
                          mode='reflect').to(device, non_blocking=True)

            # å‰å‘æŽ¨ç†
            out = self.model(x_pad)
            if isinstance(out, dict) and "pred_x" in out:
                pred = out["pred_x"]
            else:
                pred = out
            # è£å›žåŽŸå›¾å°ºå¯¸å¹¶å¤¹ç´§
            pred = torch.clamp(pred[..., :H, :W], 0.0, 1.0)  # [B,3,H,W]

            # GT
            if torch.is_tensor(y):
                gt = y.to(device, non_blocking=True)
            else:
                assert gt_root is not None, "æ— æ³•å®šä½ GT æ ¹ç›®å½•ï¼ˆè¯·æ£€æŸ¥ config.data.val_dirï¼‰"
                gts = []
                for j in range(len(y)):
                    gt_j = _load_gt_tensor(y[j], gt_root)  # [3,H,W]
                    gts.append(gt_j)
                gt = torch.stack(gts, 0).to(device)

            # å°ºå¯¸å¯¹é½ï¼ˆè£åˆ°å…±åŒæœ€å° H,Wï¼‰
            HH = min(pred.size(-2), gt.size(-2))
            WW = min(pred.size(-1), gt.size(-1))
            pred_c = pred[..., :HH, :WW]
            gt_c = gt[..., :HH, :WW]

            # è®¡ç®— PSNR
            ps = _batch_psnr(pred_c, gt_c)  # [B]
            psnr_list.append(ps.detach().cpu())

            # ==== ä¿å­˜å¢žå¼ºç»“æžœï¼ŒåŒæ—¶æž„é€ æ¯å¼ å›¾çš„æ–‡ä»¶å ====
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

            # ==== å¯¹è¿™ä¸ª batch é¢å¤–ç”Ÿæˆ PGFM / CEG ä¸­é—´å›¾ ====
            if hasattr(net, "dump_pgfm_ceg_batch"):
                net.dump_pgfm_ceg_batch(
                    x_pad.to(device),       # ä½Žå…‰è¾“å…¥ï¼ˆpad åŽï¼‰
                    pred.to(device),        # å¢žå¼ºè¾“å‡º
                    save_root,              # å½“å‰ step å­ç›®å½•
                    batch_names             # æ–‡ä»¶åå¯¹é½
                )

        psnr_avg = torch.cat(psnr_list, 0).mean().item() if len(psnr_list) else 0.0
        self.model.train()
        return psnr_avg
