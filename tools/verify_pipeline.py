#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — End-to-End Pipeline Smoke Test
#
# Verifies that:
#   1. All Python modules import cleanly.
#   2. Module A (Decoupled KFs) — three filters predict + update under PSD-constrained Q/R.
#   3. Module B (TemporalMamba) — produces PSD-valid Q/R + embedding.
#   4. Module C (Uncertainty-aware association) — cosine sim + uncertainty penalty.
#   5. Trained checkpoint (if present) loads into the inference module without strict mismatch.
#
# This script needs NO dataset and NO GPU.
#
# Usage:
#   python tools/verify_pipeline.py
#   python tools/verify_pipeline.py --ckpt checkpoints/mamba_dekf/best.pt
# ------------------------------------------------------------------------

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def step(msg: str) -> None:
    print(f"\n[step] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def assert_psd(M: torch.Tensor, name: str) -> None:
    eigs = torch.linalg.eigvalsh(M)
    if (eigs <= 0).any():
        fail(f"{name} not PSD: min eigvalue={eigs.min().item():.3e}")
    ok(f"{name} PSD (min eig={eigs.min().item():.3e}, shape={tuple(M.shape)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to a trained TemporalMamba checkpoint (optional).",
    )
    parser.add_argument("--config", default="config/train_nuscenes.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda:0 or cuda:1")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Require the configured real Mamba backend and CUDA backward")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    step("Importing modules")
    # ------------------------------------------------------------------
    from kalmanfilter.mamba_adaptive_kf import (
        PositionFilter, SizeFilter, OrientationFilter,  # noqa: F401  (import-test only)
        DecoupledAdaptiveKF, TemporalMamba, MambaDecoupledEKF,
    )
    from tracker.cost_function import (
        compute_cosine_similarity_matrix, compute_uncertainty_penalty,
    )
    ok("All Mamba-DEKF modules imported")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        fail("--require-cuda needs a CUDA GPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    model_cfg = cfg["MODEL"]
    model_kwargs = dict(
        d_model=model_cfg["D_MODEL"], d_state=model_cfg["D_STATE"],
        d_conv=model_cfg["D_CONV"], expand=model_cfg["EXPAND"],
        n_mamba_layers=model_cfg["N_MAMBA_LAYERS"], embed_dim=model_cfg["EMBED_DIM"],
        num_classes=model_cfg["NUM_CLASSES"], min_diag_q=model_cfg["MIN_DIAG_Q"],
        min_diag_r=model_cfg["MIN_DIAG_R"], min_diag_siz=model_cfg["MIN_DIAG_SIZ"],
        min_kappa=model_cfg["MIN_KAPPA"], backbone_variant=model_cfg["BACKBONE"],
        mamba2_headdim=model_cfg.get("MAMBA2_HEADDIM", 64),
        mamba2_ngroups=model_cfg.get("MAMBA2_NGROUPS", 1),
        base_noise_cfg=cfg.get("BASE_NOISE"),
        force_gru=device.type == "cpu",
    )
    print(f"  device: {device}")

    # ------------------------------------------------------------------
    step("Module A: Decoupled KFs predict+update under random PSD Q/R")
    # ------------------------------------------------------------------
    B, T = 4, model_cfg["HISTORY_LEN"]
    kf = DecoupledAdaptiveKF(batch_size=B, device=device)
    pos_x0 = torch.randn(B, 6, 1, device=device)
    siz_x0 = torch.rand(B, 3, 1, device=device) + 1.0
    ori_x0 = torch.randn(B, 2, 1, device=device)
    pos_P0 = torch.eye(6, device=device).expand(B, -1, -1).clone()
    siz_P0 = torch.eye(3, device=device).expand(B, -1, -1).clone() * 0.1
    ori_P0 = torch.eye(2, device=device).expand(B, -1, -1).clone() * 0.1
    kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)

    pos_x_p, pos_P_p, _, siz_P_p, _, ori_P_p = kf.predict(0.5)
    assert_psd(pos_P_p, "pos_P_pred")
    assert_psd(siz_P_p, "siz_P_pred")
    assert_psd(ori_P_p, "ori_P_pred")

    z_pos = torch.randn(B, 5, 1, device=device)
    z_siz = torch.rand(B, 3, 1, device=device) + 1.0
    z_ori = torch.randn(B, 1, 1, device=device)
    pos_x_u, pos_P_u, _, siz_P_u, _, ori_P_u = kf.update(z_pos, z_siz, z_ori)
    assert_psd(pos_P_u, "pos_P_upd")
    assert_psd(siz_P_u, "siz_P_upd")
    assert_psd(ori_P_u, "ori_P_upd")

    # ------------------------------------------------------------------
    step("Module B: TemporalMamba produces PSD Q/R and embedding")
    # ------------------------------------------------------------------
    mamba = TemporalMamba(**model_kwargs).to(device)
    if args.require_cuda and mamba.backbone_type != model_cfg["BACKBONE"]:
        fail(f"Requested {model_cfg['BACKBONE']}, got {mamba.backbone_type}")
    print(f"  backbone: {mamba.backbone_type} (CPU fallback checks do not validate CUDA kernels)")
    history = torch.randn(B, T, 12, device=device)
    out = mamba(history, mode=cfg["FILTER_MODE"], apply_force_prior=False)
    for k in ("Q_pos", "Q_siz", "Q_ori", "R_pos", "R_siz", "R_ori"):
        assert_psd(out[k], k)
    if out["embedding"].shape != (B, model_cfg["EMBED_DIM"]):
        fail(f"unexpected embedding shape {out['embedding'].shape}")
    ok(f"embedding shape {tuple(out['embedding'].shape)}")
    backward_loss = sum(value.square().mean() for value in out.values()
                        if torch.is_tensor(value) and value.requires_grad)
    backward_loss.backward()
    grads = [p.grad for p in mamba.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(grad).all().item() for grad in grads):
        fail("Nonfinite or missing model gradients")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ok("Configured backbone forward + backward")

    # ------------------------------------------------------------------
    step("Full pipeline: MambaDecoupledEKF.predict_with_mamba")
    # ------------------------------------------------------------------
    pipe = MambaDecoupledEKF(batch_size=B, device=device, **model_kwargs).to(device)
    pipe.kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)
    mout, px, pP, sx, sP, ox, oP = pipe.predict_with_mamba(
        history, delta_t=0.5, mode=cfg["FILTER_MODE"], apply_force_prior=False)
    assert_psd(pP, "pos_P (full pipeline)")
    assert_psd(sP, "siz_P (full pipeline)")
    assert_psd(oP, "ori_P (full pipeline)")
    pipe.update_with_mamba(z_pos, z_siz, z_ori, mout)
    ok("predict_with_mamba + update_with_mamba")

    # ------------------------------------------------------------------
    step("Module C: cosine similarity + uncertainty penalty")
    # ------------------------------------------------------------------
    emb_trk = mout["embedding"].detach().cpu().numpy()
    emb_det = np.random.randn(5, model_cfg["EMBED_DIM"]).astype(np.float32)
    sim = compute_cosine_similarity_matrix(emb_trk, emb_det)
    if sim.shape != (B, 5):
        fail(f"cosine sim shape {sim.shape}, expected ({B}, 5)")
    if (sim < -1 - 1e-4).any() or (sim > 1 + 1e-4).any():
        fail(f"cosine sim out of [-1,1]: range [{sim.min():.3f}, {sim.max():.3f}]")
    ok(f"cosine sim shape {sim.shape}, range [{sim.min():.3f}, {sim.max():.3f}]")

    pen = compute_uncertainty_penalty(
        [pP[i].detach().cpu().numpy() for i in range(B)],
        [oP[i].detach().cpu().numpy() for i in range(B)],
    )
    if pen.shape != (B,) or (pen < 0).any():
        fail(f"uncertainty penalty bad: shape={pen.shape}, min={pen.min()}")
    ok(f"uncertainty penalty shape {pen.shape}, min={pen.min():.3f}, max={pen.max():.3f}")

    # ------------------------------------------------------------------
    step("Checkpoint loading (if available)")
    # ------------------------------------------------------------------
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            fail(f"Checkpoint does not exist: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        # strip stale _tril_rows/_tril_cols keys from old checkpoints
        for k in list(sd.keys()):
            if "_tril_rows" in k or "_tril_cols" in k:
                del sd[k]
        mamba.load_state_dict(sd, strict=True)
        ok(f"Loaded checkpoint from {args.ckpt}")
        if "epoch" in ckpt:
            print(f"  trained epochs: {ckpt['epoch']}")
        if "val_loss" in ckpt:
            print(f"  val_loss: {ckpt['val_loss']}")
    else:
        print("  [info] no --ckpt supplied — skipping load test "
              "(this is fine if you have not finished training).")

    print("\n[done] Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
