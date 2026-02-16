import torch
import torch.nn as nn
import torchvision.models as models

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import os

data_root = os.path.join("images")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


from models import MicrostructureNet
from datasets import MicrostructurePatchDataset
from utils import train_single_val
from utils import (
                    evaluate_and_visualize_single_head,
                    plot_misclassified,
                    create_heatmaps_per_image,
                    create_misclassification_density_maps,
                    create_per_class_error_maps,
                    create_uncertainty_maps
                )
from utils import make_cv_loaders, make_train_val_test_loaders
from utils import StoreParams

BATCH_SIZE=128
PATCH_SIZE=128
STRIDE=64
LBP_SETTINGS=None
LR = 1e-3
EPOCHS = 50

def main():
    # lbp_settings=[(16, 2), (24,3), (32, 4)]
    print("Hello from bainite-martensite!")
    train_loader, val_loader, test_loader = make_train_val_test_loaders(
        data_root,
        batch_size=BATCH_SIZE,
        patch_size=256,
        stride=64,
        lbp_settings=LBP_SETTINGS,
        val_frac=0.15,
        test_frac=0.15,
        num_workers=0, # Windows Jupyter safe - multiprocessing issue
        random_state=42
    )


    param_tracker = StoreParams()
    param_tracker.add("batch_size", BATCH_SIZE)
    param_tracker.add("patch_size", PATCH_SIZE)
    param_tracker.add("stride", STRIDE)
    param_tracker.add("LBP", LBP_SETTINGS)
    param_tracker.add("lr", LR)
    param_tracker.add("epochs", EPOCHS)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # ------------------ Model ------------------
    model = MicrostructureNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True).to(device)

    # --------------- Loss + Optimizer-----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    
    # for fold, train_loader, val_loader in make_cv_loaders

    model, history = train_single_val(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        num_epochs=EPOCHS,
        patience=7,
    )
    
    param_tracker.add("epochs", len(history))

    missclassified, coords, probs, all_eval_paths, preds, trues, prob_vectors, report, cm = evaluate_and_visualize_single_head(
        model=model,
        loader=val_loader,
        device=device,
        max_show=len(val_loader.dataset)
    )
    
    param_tracker.add_classification_report("classification_report", report)
    result_path = param_tracker.save_params()

    # 1️⃣ Confidence heatmaps
    create_heatmaps_per_image(result_path,all_eval_paths, coords, probs, sigma=2)

    # 2️⃣ Misclassified patches
    plot_misclassified(result_path, missclassified)

    # 3️⃣ Misclassification density
    create_misclassification_density_maps(result_path,all_eval_paths, coords, preds, trues)

    # 4️⃣ Per-class error maps (example target_class = 0)
    create_per_class_error_maps(result_path, all_eval_paths, coords, preds, trues, target_class=0)

    # 5️⃣ Uncertainty maps
    create_uncertainty_maps(result_path, all_eval_paths, coords, prob_vectors)

if __name__ == "__main__":
    main()
