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
    all_prob_vectors = []
    misclassified = []
    all_eval_paths = []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        labels = batch["label"].to(device)
        coords = batch["coords"].to(device)  # x1,y1,x2,y2
        img_paths = batch["img_path"]

        lbp = batch.get("lbp", None)
        if lbp is not None:
            lbp = lbp.to(device, non_blocking=True)
            logits = model(rgb, lbp)
        else:
            logits = model(rgb)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(rgb.size(0)):
            all_true.append(labels[i].item())
            all_pred.append(preds[i].item())
            all_dominant_probs.append(probs[i].cpu()[preds[i]].item())
            all_coords.append(coords[i].cpu().tolist())
            all_prob_vectors.append(probs[i].cpu().numpy())
            all_eval_paths.append(img_paths[i])

            if preds[i] != labels[i] and len(misclassified) < max_show:
                misclassified.append({
                    "image": rgb[i].cpu(),
                    "img_path" : img_paths[i],
                    "true": labels[i].item(),
                    "pred": preds[i].item(),
                    "prob_pred": probs[i].cpu()[preds[i]].item(),
                    "coords": coords[i].cpu().tolist()
                })

    return misclassified, all_coords, all_dominant_probs, all_eval_paths, all_pred, all_true, all_prob_vectors
