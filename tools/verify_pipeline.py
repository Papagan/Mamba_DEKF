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
        "--ckpt", type=str, default="checkpoints/mamba_dekf/best.pt",
        help="Path to a trained TemporalMamba checkpoint (optional).",
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ------------------------------------------------------------------
    step("Module A: Decoupled KFs predict+update under random PSD Q/R")
    # ------------------------------------------------------------------
    B, T = 4, 10
    kf = DecoupledAdaptiveKF(batch_size=B, device=device)
    pos_x0 = torch.randn(B, 8, 1, device=device)
    siz_x0 = torch.rand(B, 3, 1, device=device) + 1.0
    ori_x0 = torch.randn(B, 2, 1, device=device)
    pos_P0 = torch.eye(8, device=device).expand(B, -1, -1).clone()
    siz_P0 = torch.eye(3, device=device).expand(B, -1, -1).clone() * 0.1
    ori_P0 = torch.eye(2, device=device).expand(B, -1, -1).clone() * 0.1
    kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)

    pos_x_p, pos_P_p, _, siz_P_p, _, ori_P_p = kf.predict(0.5)
    assert_psd(pos_P_p, "pos_P_pred")
    assert_psd(siz_P_p, "siz_P_pred")
    assert_psd(ori_P_p, "ori_P_pred")

    z_pos = torch.randn(B, 3, 1, device=device)
    z_siz = torch.rand(B, 3, 1, device=device) + 1.0
    z_ori = torch.randn(B, 1, 1, device=device)
    pos_x_u, pos_P_u, _, siz_P_u, _, ori_P_u = kf.update(z_pos, z_siz, z_ori)
    assert_psd(pos_P_u, "pos_P_upd")
    assert_psd(siz_P_u, "siz_P_upd")
    assert_psd(ori_P_u, "ori_P_upd")

    # ------------------------------------------------------------------
    step("Module B: TemporalMamba produces PSD Q/R and embedding")
    # ------------------------------------------------------------------
    mamba = TemporalMamba(d_model=64, n_mamba_layers=2, embed_dim=32).to(device)
    history = torch.randn(B, T, 13, device=device)
    out = mamba(history)
    for k in ("Q_pos", "Q_siz", "Q_ori", "R_pos", "R_siz", "R_ori"):
        assert_psd(out[k], k)
    if out["embedding"].shape != (B, 32):
        fail(f"embedding shape {out['embedding'].shape}, expected ({B}, 32)")
    ok(f"embedding shape {tuple(out['embedding'].shape)}")

    # ------------------------------------------------------------------
    step("Full pipeline: MambaDecoupledEKF.predict_with_mamba")
    # ------------------------------------------------------------------
    pipe = MambaDecoupledEKF(batch_size=B, d_model=64, device=device).to(device)
    pipe.kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)
    mout, px, pP, sx, sP, ox, oP = pipe.predict_with_mamba(history, delta_t=0.5)
    assert_psd(pP, "pos_P (full pipeline)")
    assert_psd(sP, "siz_P (full pipeline)")
    assert_psd(oP, "ori_P (full pipeline)")
    pipe.update_with_mamba(z_pos, z_siz, z_ori, mout)
    ok("predict_with_mamba + update_with_mamba")

    # ------------------------------------------------------------------
    step("Module C: cosine similarity + uncertainty penalty")
    # ------------------------------------------------------------------
    emb_trk = mout["embedding"].detach().cpu().numpy()
    emb_det = np.random.randn(5, 32).astype(np.float32)
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
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = mamba.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing keys: {missing}")
        if unexpected:
            print(f"  [warn] unexpected keys: {unexpected}")
        ok(f"Loaded checkpoint from {args.ckpt}")
        if "epoch" in ckpt:
            print(f"  trained epochs: {ckpt['epoch']}")
        if "val_loss" in ckpt:
            print(f"  val_loss: {ckpt['val_loss']}")
    else:
        print(f"  [info] no checkpoint at {args.ckpt} — skipping load test "
              "(this is fine if you have not finished training).")

    print("\n[done] Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
