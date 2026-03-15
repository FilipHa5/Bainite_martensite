import torch
import os
from models import MicrostructureNet, MicrostructureDenseNet
# from utils import run_nested_cv
from save_params import StoreParams
from loaders import make_cv_loaders
from plot_training import plot_training_history
from torch import nn
from train import train_single_val
from nn_cv_trainer import run_nested_cv_dual_separate, run_outer_fold
import numpy as np
data_root = os.path.join("images")

BATCH_SIZE=128
PATCH_SIZE=128
STRIDE=64
LBP_SETTINGS=None#[(16, 2)]
LR = 1e-3
EPOCHS = 100


def main():
    print("Hello from bainite-martensite!")

    # Store parameters and result path
    param_tracker = StoreParams()
    param_tracker.add("batch_size", BATCH_SIZE)
    param_tracker.add("patch_size", PATCH_SIZE)
    param_tracker.add("stride", STRIDE)
    param_tracker.add("LBP", LBP_SETTINGS)
    param_tracker.add("lr", LR)
    param_tracker.add("epochs", EPOCHS)
    

    # -------------------------
    # Model builders
    # -------------------------
    def build_resnet_model():
        return MicrostructureNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True)
    
    def build_densenet_model():
        return MicrostructureDenseNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True)
        
    def build_dt_model():
        raise NotImplementedError

    # -------------------------
    # Detect if running as single outer fold (parallel)
    # -------------------------
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_fold", type=int, required=False, default=None)
    parser.add_argument("--results", type=str, required=False, default="results")
    args = parser.parse_args()

    result_path = param_tracker.save_params(args.results)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.outer_fold is not None:
        # -------------------------
        # Run SINGLE outer fold (for SLURM array or parallel execution)
        # -------------------------
        nn_score, dt_score = run_outer_fold(
            outer_fold=args.outer_fold,
            outer_splits=4,
            result_path=result_path,
            build_primary_model=build_densenet_model,
            build_secondary_model=build_resnet_model,
            data_root=data_root,
            inner_splits=3,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            lbp_settings=LBP_SETTINGS,
            param_grid=None,
            device=device,
            epochs=EPOCHS
        )

        print(f"\nFold {args.outer_fold} finished.")
        print("NN score:", nn_score)
        print("DT score:", dt_score)

    else:
        # -------------------------
        # Run all outer folds sequentially
        # -------------------------
        print("Running all outer folds sequentially...")
        outer_scores_nn = []
        outer_scores_dt = []
        for fold in range(4):
            nn_score, dt_score = run_outer_fold(
                outer_fold=fold,
                outer_splits=4,
                result_path=result_path,
                build_primary_model=build_resnet_model,
                build_secondary_model=build_dt_model,
                data_root=data_root,
                inner_splits=3,
                batch_size=BATCH_SIZE,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
                lbp_settings=LBP_SETTINGS,
                param_grid=None,
                device=device,
                epochs=EPOCHS
            )
            outer_scores_nn.append(nn_score)
            outer_scores_dt.append(dt_score)
            print(f"Fold {fold} finished. NN: {nn_score}, DT: {dt_score}")

        print("\n===== FINAL NESTED CV RESULT =====")
        print("Mean NN:", np.mean(outer_scores_nn), "Std NN:", np.std(outer_scores_nn))
        print("Mean DT:", np.mean(outer_scores_dt), "Std DT:", np.std(outer_scores_dt))


if __name__ == "__main__":
    main()