import argparse
import os
import numpy as np
import torch

from models import MicrostructureResNet50, MicrostructureDenseNet
from save_params import StoreParams
from nn_cv_trainer import run_outer_fold


# =====================================================
# Configuration
# =====================================================

DATA_ROOT = "images"

BATCH_SIZE = 128
PATCH_SIZE = 128
STRIDE = 64
LBP_SETTINGS = [(24, 3)]

LR = 1e-3
EPOCHS = 100

OUTER_SPLITS = 4
INNER_SPLITS = 3

SEEDS = range(24)


# =====================================================
# Model builders
# =====================================================

def build_resnet50():
    return MicrostructureResNet50(
        lbp_settings=LBP_SETTINGS,
        freeze_backbone=True,
    )


def build_densenet():
    return MicrostructureDenseNet(
        lbp_settings=LBP_SETTINGS,
        freeze_backbone=True,
    )


def build_dt():
    raise NotImplementedError


# =====================================================
# Utilities
# =====================================================

def create_result_path_and_save_params(results_dir):
    tracker = StoreParams(base_dir=results_dir)
    tracker.add("batch_size", BATCH_SIZE)
    tracker.add("patch_size", PATCH_SIZE)
    tracker.add("stride", STRIDE)
    tracker.add("LBP", LBP_SETTINGS)
    tracker.add("epochs", EPOCHS)
    tracker.add("seed", seed)

    return tracker, tracker.get_dir()


def run_fold(fold, result_path, device, seed, secondary_type="dt"):
    secondary_builder = (
        build_densenet if secondary_type == "xgb" else build_dt
    )

    return run_outer_fold(
        outer_fold=fold,
        outer_splits=OUTER_SPLITS,
        result_path=result_path,
        build_primary_model=build_densenet,
        build_secondary_model=secondary_builder,
        data_root=DATA_ROOT,
        inner_splits=INNER_SPLITS,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        lbp_settings=LBP_SETTINGS,
        param_grid=None,
        device=device,
        epochs=EPOCHS,
        secondary_type=secondary_type,
        seed=seed,
    )


def run_all_folds(result_path, device, seed):
    nn_scores = []
    dt_scores = []

    for fold in range(OUTER_SPLITS):
        nn, dt = run_fold(
            fold=fold,
            result_path=result_path,
            device=device,
            seed=seed,
        )

        nn_scores.append(nn)
        dt_scores.append(dt)

        print(f"Fold {fold}: NN={nn:.4f}, DT={dt:.4f}")

    print("\n===== FINAL NESTED CV RESULT =====")
    print(f"NN : {np.mean(nn_scores):.4f} ± {np.std(nn_scores):.4f}")
    print(f"DT : {np.mean(dt_scores):.4f} ± {np.std(dt_scores):.4f}")


# =====================================================
# Main
# =====================================================

def main(seed, args):
    print(f"\n========== Seed {seed} ==========")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_tracker, result_path = create_result_path_and_save_params(args.results)

    print("Using device:", device)

    if args.outer_fold is not None:
        nn, dt = run_fold(
            fold=args.outer_fold,
            result_path=result_path,
            device=device,
            seed=seed,
            secondary_type="xgb",
        )

        param_tracker.add("primary_model_score", nn)
        param_tracker.add("secondary_model_score", dt)

        print(f"\nFold {args.outer_fold} finished.")
        print("Seed:", seed)
        print("NN score:", nn)
        print("DT score:", dt)

    else:
        run_all_folds(result_path, device, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_fold", type=int, default=None)
    parser.add_argument("--results", type=str, default="results")

    args = parser.parse_args()

    for seed in SEEDS:
        main(seed, args)