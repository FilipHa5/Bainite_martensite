
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from itertools import product
import torch
from torch import nn
from datasets import MicrostructurePatchDataset
from train import train_single_val, train_full
from evaluation import (
                    evaluate_and_visualize_single_head,
                    plot_misclassified,
                    create_heatmaps_per_image,
                    create_misclassification_density_maps,
                    create_per_class_error_maps,
                    create_uncertainty_maps,
                    perform_all_patches_corrections
                )
from plot_training import plot_training_history
import numpy as np
from models import train_dt_model
from loaders import make_cv_loaders_from_samples, collect_img_paths_with_labels, make_loader_from_samples
import os

def run_nested_cv_dual_separate(
    result_path,
    build_primary_model,
    build_secondary_model,
    data_root,
    outer_splits=4,
    inner_splits=3,
    batch_size=128,
    patch_size=256,
    stride=64,
    lbp_settings=[(1, 8), (3, 16)],
    param_grid=None,
    device="cuda",
    epochs=50
):

    """
    Nested CV tuning both:
    - Primary classifier (NN)
    - Secondary classifier (DT/LBP)
    Separately.
    """

    if param_grid is None:
        param_grid = {
            "lr": [1e-3, 1e-4],
            "weight_decay": [0, 1e-4],
            "bins": [8, 16, 32, 64, 128, 256],
        }

    # Load all samples
    all_samples = collect_img_paths_with_labels(data_root)
    y = np.array([s[1] for s in all_samples])

    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

    outer_scores_nn = []
    outer_scores_dt = []

    criterion = nn.CrossEntropyLoss()
    # -------------------------
    # OUTER LOOP
    # -------------------------
    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_kf.split(all_samples, y)):
        print(f"\n========== OUTER FOLD {outer_fold} ==========")

        result_path = f"{result_path}_OF_{outer_fold}"
        os.makedirs(result_path, exist_ok=True)

        trainval_samples = [all_samples[i] for i in trainval_idx]
        test_samples = [all_samples[i] for i in test_idx]
        y_trainval = y[trainval_idx]

        best_params_nn = {}
        best_score_nn = -np.inf

        best_params_dt = {}
        best_score_dt = -np.inf

        # -------------------------
        # INNER LOOP for NN
        # -------------------------
        for lr, wd in product(param_grid["lr"], param_grid["weight_decay"]):
            print(f"Inner loop NN, LR: {lr}, WD: {wd}")
            inner_scores_nn = []
            inner_best_epochs = []

            for _, train_loader, val_loader in make_cv_loaders_from_samples(
                trainval_samples,
                batch_size=batch_size,
                n_splits=inner_splits,
                patch_size=patch_size,
                stride=stride,
                lbp_settings=lbp_settings,
            ):

                nn_model = build_primary_model().to(device)
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=wd)

                model, history = train_single_val(
                    nn_model,
                    optimizer,
                    criterion,
                    train_loader,
                    val_loader,
                    device=device,
                    num_epochs=epochs,
                    patience=7,
                )
                score = max(history["val_acc"])
                inner_scores_nn.append(score)

                best_epoch_fold = np.argmax(history["val_acc"]) + 1
                inner_best_epochs.append(best_epoch_fold)

            mean_score_nn = np.mean(inner_scores_nn)
            mean_epoch_nn = int(np.median(inner_best_epochs))
            if mean_score_nn > best_score_nn:
                best_score_nn = mean_score_nn
                best_params_nn = {
                    "lr": lr,
                    "weight_decay": wd,
                    "epochs": mean_epoch_nn
    }

        # -------------------------
        # INNER LOOP for DT
        # -------------------------
        for bins in param_grid["bins"]:
            print("Inner loop, DT bins", bins)
            inner_scores_dt = []

            for _, train_loader, val_loader in make_cv_loaders_from_samples(
                trainval_samples,
                batch_size=batch_size,
                n_splits=inner_splits,
                patch_size=patch_size,
                stride=stride,
                lbp_settings=lbp_settings,
            ):

                dt_model, score_dt = train_dt_model(train_loader, val_loader, bins)
                inner_scores_dt.append(score_dt)

            mean_score_dt = np.mean(inner_scores_dt)
            if mean_score_dt > best_score_dt:
                best_score_dt = mean_score_dt
                best_params_dt = {"bins": bins}

        print("Best inner params NN:", best_params_nn)
        print("Best inner params DT:", best_params_dt, "accuracy score: ", score_dt)

        # -------------------------
        # FINAL TRAIN ON FULL TRAINVAL
        # -------------------------
        full_train_loader = make_loader_from_samples(trainval_samples, patch_size, stride, True, lbp_settings)
        test_loader = make_loader_from_samples(test_samples, patch_size, stride, False, lbp_settings, shuffle=False)


        # ---- Train NN with best params ----
        nn_model = build_primary_model().to(device)
        optimizer = torch.optim.Adam(
            nn_model.parameters(),
            lr=best_params_nn["lr"],
            weight_decay=best_params_nn["weight_decay"],
        )
        num_epochs = best_params_nn["epochs"]
        print(f"Training NN on full data: {num_epochs} epochs")
        train_full(nn_model, optimizer, criterion, full_train_loader, device, num_epochs=num_epochs)

        # ---- Train DT with best params ----
        dt_classifier, dt_accuracy = train_dt_model(full_train_loader, test_loader, best_params_dt["bins"])
        print("Final test DT accuracy:", dt_accuracy)

        # ---- Evaluate NN ----
        (
        misclassified,
        coords,
        probs,
        all_eval_paths,
        preds,
        trues,
        prob_vectors,
        report,
        cm,
        test_accuracy_nn,
        ) = evaluate_and_visualize_single_head(
            model=nn_model,
            loader=test_loader,
            device=device,
            max_show=len(test_loader.dataset),
        )
        
        
        # 1️⃣ Confidence heatmaps
        create_heatmaps_per_image(result_path,all_eval_paths, coords, probs, sigma=2)

        # 2️⃣ Misclassified patches
        plot_misclassified(result_path, misclassified, dt_classifier)

        # 3️⃣ Misclassification density
        create_misclassification_density_maps(result_path,all_eval_paths, coords, preds, trues)

        # 4️⃣ Per-class error maps (example target_class = 0)
        create_per_class_error_maps(result_path, all_eval_paths, coords, preds, trues, target_class=0)

        # 5️⃣ Uncertainty maps
        create_uncertainty_maps(result_path, all_eval_paths, coords, prob_vectors)

        print("Outer test score NN:", test_accuracy_nn)

        outer_scores_nn.append(test_accuracy_nn)

        print("Outer test score DT:", dt_accuracy)
        outer_scores_dt.append(dt_accuracy)

    print("\n===== FINAL NESTED CV RESULT =====")
    print("Mean NN:", np.mean(outer_scores_nn), "Std NN:", np.std(outer_scores_nn))
    print("Mean DT:", np.mean(outer_scores_dt), "Std DT:", np.std(outer_scores_dt))

    return outer_scores_nn, outer_scores_dt  # optionally return DT scores too