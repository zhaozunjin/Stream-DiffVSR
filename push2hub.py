"""
train_stream_diffvsr.py
=======================
Stream-DiffVSR 完整三阶段训练脚本（适配本地目录结构版）

目录结构（与 find 输出完全一致）:
    <repo_root>/
        unet/           config.json + diffusion_pytorch_model.safetensors
        controlnet/     config.json + diffusion_pytorch_model.safetensors
        vae/            config.json + diffusion_pytorch_model.safetensors
        text_encoder/   config.json + model.safetensors
        tokenizer/
        scheduler/      scheduler_config.json  ddim_scheduler.py
        pipeline/       stream_diffvsr_pipeline.py
        temporal_autoencoder/  autoencoder_tiny.py  vae.py
        util/           flow_utils.py

运行示例（单卡）:
    cd <repo_root>
    python train.py \
        --repo_root . \
        --data_root datasets/REDS4\
        --output_dir ./checkpoints \
        --stage all

运行示例（accelerate 多卡）:
    accelerate launch --num_processes 4 train_stream_diffvsr.py \
        --repo_root . \
        --data_root /data/vsr_train \
        --output_dir ./checkpoints \
        --stage all \
        --mixed_precision bf16

数据目录格式:
    data_root/
        seq001/
            lr/  frame_0001.png  frame_0002.png ...
            hr/  frame_0001.png  frame_0002.png ...
        seq002/ ...
"""

import os
import sys
import logging
import argparse
import itertools
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

logger = get_logger(__name__, log_level="INFO")


# ============================================================
# 0.  sys.path 注入（必须在 import 仓库模块前完成）
# ============================================================
def _setup_sys_path(repo_root: str) -> str:
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


# ============================================================
# 1.  模型加载（全部从本地子目录，无需网络）
# ============================================================
def load_models(repo_root: str):
    """加载全部四个组件，返回 (unet, controlnet, vae, scheduler)。"""
    from pipeline.stream_diffvsr_pipeline import (
        ControlNetModel, UNet2DConditionModel,
    )
    from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

    unet       = UNet2DConditionModel.from_pretrained(  os.path.join(repo_root, "unet"), use_safetensors=True)
    controlnet = ControlNetModel.from_pretrained(       os.path.join(repo_root, "controlnet"), use_safetensors=True)
    vae        = TemporalAutoencoderTiny.from_pretrained(os.path.join(repo_root, "vae"), use_safetensors=True)
    scheduler  = DDIMScheduler.from_pretrained(         os.path.join(repo_root, "scheduler"),use_safetensors=True)
    return unet, controlnet, vae, scheduler


def _best_path(ckpt_dir: Optional[str], subdir: str, fallback: str) -> str:
    if ckpt_dir:
        p = os.path.join(ckpt_dir, subdir)
        if os.path.exists(p):
            return p
    return fallback


def load_unet_controlnet(repo_root: str, ckpt_dir: Optional[str] = None):
    from pipeline.stream_diffvsr_pipeline import ControlNetModel, UNet2DConditionModel
    return (
        UNet2DConditionModel.from_pretrained(
            _best_path(ckpt_dir, "unet",       os.path.join(repo_root, "unet"))),
        ControlNetModel.from_pretrained(
            _best_path(ckpt_dir, "controlnet",  os.path.join(repo_root, "controlnet"))),
    )


def load_vae(repo_root: str, ckpt_dir: Optional[str] = None):
    from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
    return TemporalAutoencoderTiny.from_pretrained(
        _best_path(ckpt_dir, "vae", os.path.join(repo_root, "vae")))


# ============================================================
# 2.  感知损失（VGG16，不依赖 lpips 包）
# ============================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23],
        ])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 输入 [-1,1] → [0,1] → ImageNet-normalize
        pred   = ((pred   * 0.5 + 0.5).clamp(0,1) - self.mean) / self.std
        target = ((target * 0.5 + 0.5).clamp(0,1) - self.mean) / self.std
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for blk in self.blocks:
            x = blk(x); y = blk(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# ============================================================
# 3.  Dataset
# ============================================================
# class VSRVideoDataset(Dataset):
#     def __init__(self, data_root, clip_len=5,
#                  lr_size=(180,320), hr_size=(720,1280), augment=True):
#         self.clip_len = clip_len
#         self.augment  = augment
#         self.lr_size  = lr_size
#         self.hr_size  = hr_size
#         self.clips: List[Tuple] = []

#         root = Path(data_root)
#         for seq_dir in sorted(root.iterdir()):
#             lr_dir = seq_dir / "lr"
#             hr_dir = seq_dir / "hr"
#             if not lr_dir.exists() or not hr_dir.exists():
#                 continue
#             lr_frames = sorted(list(lr_dir.glob("*.png")) + list(lr_dir.glob("*.jpg")))
#             hr_frames = sorted(list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg")))
#             if len(lr_frames) != len(hr_frames) or len(lr_frames) < clip_len:
#                 continue
#             for s in range(len(lr_frames) - clip_len + 1):
#                 self.clips.append((lr_frames[s:s+clip_len], hr_frames[s:s+clip_len]))

#     def __len__(self): return len(self.clips)

#     def _load(self, path: Path, size: Tuple[int,int]) -> torch.Tensor:
#         img = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BICUBIC)
#         return TF.to_tensor(img)   # [0,1]

#     def __getitem__(self, idx):
#         lr_ps, hr_ps = self.clips[idx]
#         lrs = [self._load(p, self.lr_size) for p in lr_ps]
#         hrs = [self._load(p, self.hr_size) for p in hr_ps]
#         if self.augment and torch.rand(1).item() > 0.5:
#             lrs = [TF.hflip(f) for f in lrs]
#             hrs = [TF.hflip(f) for f in hrs]
#         lr = torch.stack(lrs) * 2.0 - 1.0   # [T,3,H,W] → [-1,1]
#         hr = torch.stack(hrs) * 2.0 - 1.0
#         return {"lr": lr, "hr": hr}


class VSRVideoDataset(Dataset):
    def __init__(self, data_root, clip_len=5,
                 lr_size=(180,320), hr_size=(720,1280), augment=True):
        self.clip_len = clip_len
        self.augment  = augment
        self.lr_size  = lr_size
        self.hr_size  = hr_size
        self.clips: List[Tuple] = []

        root = Path(data_root)

        # ✅ 直接定位 BIx4 和 GT
        lr_root = root / "BIx4"
        hr_root = root / "GT"

        if not lr_root.exists() or not hr_root.exists():
            raise ValueError(f"Cannot find BIx4 / GT under {data_root}")

        # ✅ 遍历 scene（walk, calendar, ...）
        for scene_dir in sorted(lr_root.iterdir()):
            if not scene_dir.is_dir():
                continue

            scene_name = scene_dir.name
            hr_scene_dir = hr_root / scene_name

            if not hr_scene_dir.exists():
                continue

            lr_frames = sorted(
                list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.jpg"))
            )
            hr_frames = sorted(
                list(hr_scene_dir.glob("*.png")) + list(hr_scene_dir.glob("*.jpg"))
            )

            # ✅ 必须一一对应
            if len(lr_frames) != len(hr_frames):
                continue

            if len(lr_frames) < clip_len:
                continue

            # ✅ 构造 clip
            for s in range(len(lr_frames) - clip_len + 1):
                self.clips.append((
                    lr_frames[s:s + clip_len],
                    hr_frames[s:s + clip_len]
                ))

        print(f"[Dataset] Total clips: {len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def _load(self, path: Path, size: Tuple[int,int]) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize(
            (size[1], size[0]), Image.BICUBIC
        )
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        lr_ps, hr_ps = self.clips[idx]

        lrs = [self._load(p, self.lr_size) for p in lr_ps]
        hrs = [self._load(p, self.hr_size) for p in hr_ps]

        # 简单翻转增强
        if self.augment and torch.rand(1).item() > 0.5:
            lrs = [TF.hflip(f) for f in lrs]
            hrs = [TF.hflip(f) for f in hrs]

        lr = torch.stack(lrs) * 2.0 - 1.0  # [-1,1]
        hr = torch.stack(hrs) * 2.0 - 1.0

        return {"lr": lr, "hr": hr}
    
def collate_fn(batch):
    return {
        "lr": torch.stack([b["lr"] for b in batch]),
        "hr": torch.stack([b["hr"] for b in batch]),
    }


# ============================================================
# 4.  VAE 编解码辅助
# ============================================================
_VAE_SCALE = 0.18215


def encode_latents(vae, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = vae.encode(x)
    if hasattr(out, "latents"):
        z = out.latents
    elif hasattr(out, "latent_dist"):
        z = out.latent_dist.sample()
    else:
        z = out
    return z * _VAE_SCALE


def decode_latents(vae, z: torch.Tensor, f_prev=None) -> torch.Tensor:
    z = z / _VAE_SCALE
    try:
        out = vae.decode(z, f_prev=f_prev) if f_prev is not None else vae.decode(z)
    except TypeError:
        out = vae.decode(z)
    return out.sample if hasattr(out, "sample") else out


# ============================================================
# 5.  去噪辅助
# ============================================================
def add_noise_at_t(z_gt, t_int, scheduler):
    noise    = torch.randn_like(z_gt)
    t_tensor = torch.tensor([t_int], dtype=torch.long, device=z_gt.device)
    return scheduler.add_noise(z_gt, noise, t_tensor), noise


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def denoise_loop(unet, controlnet, scheduler,
                 z_noisy, x_cond, timesteps: List[int],
                 device, dtype) -> torch.Tensor:
    """
    x_cond : [B, C, H, W]  ControlNet 条件图（Stage1: x_lq 3ch；Stage3: concat 6ch）
    """
    z   = z_noisy
    B   = z.shape[0]
    dim = unet.config.cross_attention_dim
    enc = torch.zeros(B, 1, dim, device=device, dtype=dtype)

    is_training = unet.training or (controlnet is not None and controlnet.training)
    ctx = _nullctx() if is_training else torch.no_grad()

    with ctx:
        for t_int in timesteps:
            t = torch.tensor([t_int] * B, dtype=torch.long, device=device)
            if controlnet is not None:
                down_res, mid_res = controlnet(
                    z, t,
                    encoder_hidden_states=enc,
                    controlnet_cond=x_cond,
                    return_dict=False,
                )
                noise_pred = unet(
                    z, t,
                    encoder_hidden_states=enc,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample
            else:
                noise_pred = unet(z, t, encoder_hidden_states=enc).sample
            z = scheduler.step(noise_pred, t[0], z).prev_sample
    return z


# ============================================================
# 6.  存档
# ============================================================
def save_checkpoint(accelerator, unet, controlnet, vae, out_dir, tag):
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    if unet       is not None: accelerator.unwrap_model(unet).save_pretrained(      os.path.join(d, "unet"))
    if controlnet is not None: accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(d, "controlnet"))
    if vae        is not None: accelerator.unwrap_model(vae).save_pretrained(       os.path.join(d, "vae"))
    logger.info(f"Checkpoint saved → {d}")


# ============================================================
# 7.  Stage 1：训练 UNet + ControlNet
# ============================================================
def stage1_train_unet(args, accelerator: Accelerator):
    logger.info("=" * 60)
    logger.info("STAGE 1: Training UNet + ControlNet denoiser")
    logger.info("=" * 60)

    unet, controlnet, vae, scheduler = load_models(args.repo_root)
    vae.requires_grad_(False)
    unet.train(); controlnet.train()

    perceptual = VGGPerceptualLoss()
    params     = list(unet.parameters()) + list(controlnet.parameters())
    optimizer  = torch.optim.AdamW(params, lr=args.stage1_lr,
                                   betas=(0.9,0.999), weight_decay=1e-2)

    dataset    = VSRVideoDataset(args.data_root, clip_len=args.clip_len,
                                 lr_size=tuple(args.lr_size), hr_size=tuple(args.hr_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True, drop_last=True)
    lr_sched   = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=500,
                               num_training_steps=args.stage1_steps)

    unet, controlnet, optimizer, dataloader, lr_sched, perceptual = \
        accelerator.prepare(unet, controlnet, optimizer, dataloader, lr_sched, perceptual)
    vae = vae.to(accelerator.device)

    TIMESTEPS = [999, 749, 499, 249]
    data_iter = itertools.cycle(dataloader)

    for step in range(1, args.stage1_steps + 1):
        batch   = next(data_iter)
        hr_clip = batch["hr"]              # [B,T,3,H,W]
        lr_clip = batch["lr"]
        B, T    = hr_clip.shape[:2]
        fi      = torch.randint(0, T, (1,)).item()
        x_gt    = hr_clip[:, fi]
        x_lq    = lr_clip[:, fi]

        z_gt          = encode_latents(vae, x_gt)
        z_noisy, _    = add_noise_at_t(z_gt, TIMESTEPS[0], scheduler)
        z_pred        = denoise_loop(unet, controlnet, scheduler,
                                     z_noisy, x_lq, TIMESTEPS,
                                     accelerator.device, z_noisy.dtype)

        loss_z     = F.mse_loss(z_pred, z_gt)
        x_pred     = decode_latents(vae, z_pred)
        loss_lpips = perceptual(x_pred, x_gt)
        loss       = args.lambda_z * loss_z + args.lambda_lpips * loss_lpips

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step(); lr_sched.step(); optimizer.zero_grad()

        if step % args.log_every == 0:
            logger.info(f"[Stage1] {step}/{args.stage1_steps}  "
                        f"loss={loss.item():.4f}  z={loss_z.item():.4f}  "
                        f"lpips={loss_lpips.item():.4f}  "
                        f"lr={lr_sched.get_last_lr()[0]:.2e}")

        if step % args.save_every == 0 or step == args.stage1_steps:
            save_checkpoint(accelerator, unet, controlnet, None,
                            args.output_dir, f"stage1_step{step:07d}")

    logger.info("Stage 1 finished.")
    return accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet)


# ============================================================
# 8.  Stage 2：训练 Temporal Decoder + TPM
# ============================================================
def stage2_train_temporal_decoder(args, accelerator: Accelerator,
                                  unet=None, controlnet=None):
    logger.info("=" * 60)
    logger.info("STAGE 2: Training Temporal Decoder + TPM")
    logger.info("=" * 60)

    s1_ckpt = os.path.join(args.output_dir, f"stage1_step{args.stage1_steps:07d}")
    if unet is None or controlnet is None:
        unet, controlnet = load_unet_controlnet(
            args.repo_root, s1_ckpt if os.path.exists(s1_ckpt) else None)
    vae      = load_vae(args.repo_root)
    _, _, _, scheduler = load_models(args.repo_root)

    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    # 只训练 decoder（和 TPM）；encoder 冻结
    for name, param in vae.named_parameters():
        param.requires_grad_("encoder" not in name)

    perceptual = VGGPerceptualLoss()
    params     = [p for p in vae.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(params, lr=args.stage2_lr)

    dataset    = VSRVideoDataset(args.data_root, clip_len=max(args.clip_len, 2),
                                 lr_size=tuple(args.lr_size), hr_size=tuple(args.hr_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True, drop_last=True)
    lr_sched   = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=300,
                               num_training_steps=args.stage2_steps)

    vae, optimizer, dataloader, lr_sched, perceptual = \
        accelerator.prepare(vae, optimizer, dataloader, lr_sched, perceptual)
    dev = accelerator.device
    unet = unet.to(dev); controlnet = controlnet.to(dev)

    TIMESTEPS = [999, 749, 499, 249]
    data_iter = itertools.cycle(dataloader)

    for step in range(1, args.stage2_steps + 1):
        batch   = next(data_iter)
        hr_clip = batch["hr"]; lr_clip = batch["lr"]
        B, T    = hr_clip.shape[:2]
        fi      = torch.randint(1, T, (1,)).item()
        x_gt    = hr_clip[:, fi]
        x_prev  = hr_clip[:, fi - 1]
        x_lq    = lr_clip[:, fi]

        with torch.no_grad():
            z_gt       = encode_latents(vae, x_gt)
            z_noisy, _ = add_noise_at_t(z_gt, TIMESTEPS[0], scheduler)
            z_pred     = denoise_loop(unet, controlnet, scheduler,
                                      z_noisy, x_lq, TIMESTEPS, dev, z_noisy.dtype)

        x_pred     = decode_latents(vae, z_pred, f_prev=x_prev)
        loss_mse   = F.mse_loss(x_pred, x_gt)
        loss_lpips = perceptual(x_pred, x_gt)
        loss       = args.lambda_mse * loss_mse + args.lambda_lpips * loss_lpips

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step(); lr_sched.step(); optimizer.zero_grad()

        if step % args.log_every == 0:
            logger.info(f"[Stage2] {step}/{args.stage2_steps}  "
                        f"loss={loss.item():.4f}  "
                        f"mse={loss_mse.item():.4f}  "
                        f"lpips={loss_lpips.item():.4f}")

        if step % args.save_every == 0 or step == args.stage2_steps:
            save_checkpoint(accelerator, None, None, vae,
                            args.output_dir, f"stage2_step{step:07d}")

    logger.info("Stage 2 finished.")
    return accelerator.unwrap_model(vae)


# ============================================================
# 9.  Stage 3：端到端微调 ARTG（ControlNet）
# ============================================================
def stage3_train_artg(args, accelerator: Accelerator,
                      unet=None, controlnet=None, vae=None):
    logger.info("=" * 60)
    logger.info("STAGE 3: End-to-end fine-tuning ARTG (ControlNet)")
    logger.info("=" * 60)

    s1_ckpt = os.path.join(args.output_dir, f"stage1_step{args.stage1_steps:07d}")
    s2_ckpt = os.path.join(args.output_dir, f"stage2_step{args.stage2_steps:07d}")

    if unet is None or controlnet is None:
        unet, controlnet = load_unet_controlnet(
            args.repo_root, s1_ckpt if os.path.exists(s1_ckpt) else None)
    if vae is None:
        vae = load_vae(args.repo_root, s2_ckpt if os.path.exists(s2_ckpt) else None)
    _, _, _, scheduler = load_models(args.repo_root)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.train()

    # 光流模型（RAFT）用于 ARTG 时序对齐条件
    # 若离线环境无法自动下载，可手动指定权重路径：
    #   weights = Raft_Large_Weights.DEFAULT
    #   of_model = raft_large(weights=None)
    #   of_model.load_state_dict(torch.load("/path/to/raft_large.pth"))
    of_model = None
    if not args.no_optical_flow:
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
            logger.info("RAFT optical flow model loaded.")
        except Exception as e:
            logger.warning(f"RAFT load failed ({e}). Proceeding without optical flow.")

    perceptual = VGGPerceptualLoss()
    params     = list(controlnet.parameters())
    optimizer  = torch.optim.AdamW(params, lr=args.stage3_lr)

    dataset    = VSRVideoDataset(args.data_root, clip_len=max(args.clip_len, 2),
                                 lr_size=tuple(args.lr_size), hr_size=tuple(args.hr_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True, drop_last=True)
    lr_sched   = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=300,
                               num_training_steps=args.stage3_steps)

    controlnet, optimizer, dataloader, lr_sched, perceptual = \
        accelerator.prepare(controlnet, optimizer, dataloader, lr_sched, perceptual)
    dev = accelerator.device
    unet = unet.to(dev); vae = vae.to(dev)
    if of_model is not None:
        of_model = of_model.to(dev)
        of_model.requires_grad_(False)

    TIMESTEPS = [999, 749, 499, 249]
    data_iter = itertools.cycle(dataloader)

    for step in range(1, args.stage3_steps + 1):
        batch   = next(data_iter)
        hr_clip = batch["hr"]; lr_clip = batch["lr"]
        B, T    = hr_clip.shape[:2]
        fi      = torch.randint(1, T, (1,)).item()
        x_gt    = hr_clip[:, fi]
        x_prev  = hr_clip[:, fi - 1]
        x_lq    = lr_clip[:, fi]

        # ARTG 条件：将 x_prev 与 x_lq concat 后作为 ControlNet 条件图
        # conditioning_channels 需与 controlnet/config.json 中一致：
        #   - 若 conditioning_channels == 3  → 只传 x_lq
        #   - 若 conditioning_channels == 6  → 传 concat(x_prev, x_lq)
        #   - 若 conditioning_channels == 9  → 传 concat(x_prev, x_lq, warped_prev)
        # 默认按 6ch 处理，请根据实际 config 调整。
        # artg_cond = torch.cat([x_prev, x_lq], dim=1)   # [B, 6, H, W]
        artg_cond = x_lq   # [B, 3, H, W]

        with torch.no_grad():
            z_gt       = encode_latents(vae, x_gt)
            z_noisy, _ = add_noise_at_t(z_gt, TIMESTEPS[0], scheduler)

        z_pred     = denoise_loop(unet, controlnet, scheduler,
                                  z_noisy, artg_cond, TIMESTEPS,
                                  dev, z_noisy.dtype)
        x_pred     = decode_latents(vae, z_pred, f_prev=x_prev)

        loss_z     = F.mse_loss(z_pred, z_gt.detach())
        loss_lpips = perceptual(x_pred, x_gt)
        loss       = args.lambda_z * loss_z + args.lambda_lpips * loss_lpips

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step(); lr_sched.step(); optimizer.zero_grad()

        if step % args.log_every == 0:
            logger.info(f"[Stage3] {step}/{args.stage3_steps}  "
                        f"loss={loss.item():.4f}  "
                        f"z={loss_z.item():.4f}  "
                        f"lpips={loss_lpips.item():.4f}")

        if step % args.save_every == 0 or step == args.stage3_steps:
            save_checkpoint(accelerator, None, controlnet, vae,
                            args.output_dir, f"stage3_step{step:07d}")

    logger.info("Stage 3 finished.")


# ============================================================
# 10. CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Stream-DiffVSR training script")

    # ── 路径 ──────────────────────────────────────────────────
    p.add_argument("--repo_root",  type=str, default=".",
                   help="Repo root: directory containing unet/, vae/, controlnet/, ...")
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./checkpoints")

    # ── 数据 ──────────────────────────────────────────────────
    p.add_argument("--clip_len",   type=int, default=5)
    p.add_argument("--lr_size",    type=int, nargs=2, default=[180, 320],  metavar=("H","W"))
    p.add_argument("--hr_size",    type=int, nargs=2, default=[720, 1280], metavar=("H","W"))

    # ── 阶段 ──────────────────────────────────────────────────
    p.add_argument("--stage", choices=["1","2","3","all"], default="all")

    # ── 超参 ──────────────────────────────────────────────────
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--stage1_lr",      type=float, default=5e-5)
    p.add_argument("--stage2_lr",      type=float, default=5e-5)
    p.add_argument("--stage3_lr",      type=float, default=2e-5)
    p.add_argument("--stage1_steps",   type=int,   default=100_000)
    p.add_argument("--stage2_steps",   type=int,   default=50_000)
    p.add_argument("--stage3_steps",   type=int,   default=50_000)
    p.add_argument("--lambda_z",       type=float, default=1.0)
    p.add_argument("--lambda_mse",     type=float, default=1.0)
    p.add_argument("--lambda_lpips",   type=float, default=0.5)
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    p.add_argument("--no_optical_flow", action="store_true",
                   help="Skip RAFT optical flow loading (for offline envs)")

    # ── 日志/存档 ─────────────────────────────────────────────
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--save_every",     type=int,   default=5_000)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mixed_precision",type=str,   default="fp16",
                   choices=["no","fp16","bf16"])

    return p.parse_args()


# ============================================================
# 11. 主入口
# ============================================================
def main():
    args = parse_args()
    _setup_sys_path(args.repo_root)   # ← 必须在所有仓库 import 之前

    set_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs"),
        ),
        log_with="tensorboard",
    )

    logger.info(f"Device: {accelerator.device}  "
                f"Mixed-precision: {args.mixed_precision}  "
                f"Num processes: {accelerator.num_processes}")
    logger.info(f"Repo root : {os.path.abspath(args.repo_root)}")
    logger.info(f"Output dir: {os.path.abspath(args.output_dir)}")

    unet = controlnet = vae = None

    if args.stage in ("1", "all"):
        unet, controlnet = stage1_train_unet(args, accelerator)

    if args.stage in ("2", "all"):
        vae = stage2_train_temporal_decoder(args, accelerator, unet, controlnet)

    if args.stage in ("3", "all"):
        stage3_train_artg(args, accelerator, unet, controlnet, vae)

    logger.info("All stages complete. ✓")


if __name__ == "__main__":
    main()
