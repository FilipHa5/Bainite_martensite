import torch

def custom_collate(batch):
    rgb = torch.stack([b["rgb"] for b in batch], dim=0)

    labels = torch.stack([b["label"] for b in batch], dim=0)
    coords = torch.stack([b["coords"] for b in batch], dim=0)

    img_paths = [b["img_path"] for b in batch]  # keep as list of strings

    lbp_list = [b["lbp"] for b in batch]

    # If ANY sample has None → whole batch LBP = None
    if any(l is None for l in lbp_list):
        lbp = None
    else:
        lbp = torch.stack(lbp_list, dim=0)

    return {
        "rgb": rgb,
        "lbp": lbp,
        "label": labels,
        "img_path": img_paths,
        "coords": coords,
    }

