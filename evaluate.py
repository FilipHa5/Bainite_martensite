import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

CLASS_NAMES = [
    "Upper Bainite",
    "Martensite ± Lower Bainite"
]

@torch.no_grad()
def evaluate_and_visualize_single_head(
    model,
    loader,
    device,
    max_show=16
):
    model.eval()

    all_true = []
    all_pred = []
    all_dominant_probs = []
    all_coords = []
    misclassified = []
    missed_coordinates = []
    missed_probs = []
    paths_of_missed_src_img  = []
    eval_paths = []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)
        coords = batch["coords"].to(device) # x,y x+ps, y+ps
        img_paths = batch["img_path"]
        eval_paths.extend(img_paths)

        
        lbp = batch.get("lbp", None)
        if lbp is not None:
            lbp = lbp.to(device, non_blocking=True)
            logits = model(rgb, lbp)  # allow lbp=None in model forward
        else:
            logits = model(rgb)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(rgb.size(0)):
            all_true.append(labels[i].item())
            all_pred.append(preds[i].item())
            all_dominant_probs.append(probs[i].cpu()[preds[i]].numpy())
            all_coords.append(coords[i])
            paths_of_missed_src_img.extend(img_paths[i])

            if preds[i] != labels[i] and len(misclassified) < max_show:
                missed_coordinates.append(coords[i].cpu())
                missed_probs_cpu = probs[i].cpu()[preds[i]].numpy()
                missed_probs.append(missed_probs_cpu)
                misclassified.append({
                    "image": rgb[i].cpu(),
                    "img_path" : img_paths[i],
                    "true": labels[i].item(),
                    "pred": preds[i].item(),
                    "prob_pred": missed_probs_cpu,
                    "coords": coords[i].cpu()
                    
                })

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_dominant_probs = np.array(all_dominant_probs)

    # =============================
    # Metrics
    # =============================
    print(f"Evaluated on: {set(eval_paths)}")
    acc = 100.0 * (all_true == all_pred).mean()

    print("=" * 60)
    print("VALIDATION SUMMARY — Single-Head (Deformation Ignored)")
    print("=" * 60)
    print(f"Samples: {len(all_true)}")
    print(f"Accuracy: {acc:.2f}%\n")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    print("Confusion matrix")
    print("               Pred UB   Pred Lath")
    for i, name in enumerate(CLASS_NAMES):
        print(f"True {name:18s} {cm[i,0]:8d} {cm[i,1]:10d}")
    print()

    # Classification report
    print("Classification report")
    print(
        classification_report(
            all_true,
            all_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Mean confidence
    mean_conf = all_dominant_probs.mean()
    print(f"Mean confidence (max prob): {mean_conf:.4f}")
    print("=" * 60)

    # =============================
    # Visualization
    # =============================
    if not misclassified:
        print("No misclassified samples to visualize.")
        return

    n_show = len(misclassified)
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_show:
            ax.set_visible(False)
            continue

        m = misclassified[i]
        img = m["image"].permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img)
        ax.set_title(
            f"True: {CLASS_NAMES[m['true']]}\n"
            f"Pred: {CLASS_NAMES[m['pred']]}\n"
            f"Max p={m['prob_pred'].max():.2f}",
            fontsize=8
        )
        ax.axis("off")

    plt.suptitle("Misclassified Validation Patches", fontsize=12)
    plt.tight_layout()
    plt.show()
    return misclassified, all_coords, all_dominant_probs
