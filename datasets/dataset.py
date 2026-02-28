import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
from skimage.feature import local_binary_pattern


class MicrostructurePatchDataset(Dataset):
    def __init__(
        self,
        samples,
        patch_size=256,
        stride=64,
        augment=False,
        lbp_settings=None  # list of [P, R] tuples, e.g., [(8,1),(16,2)]
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.lbp_settings = lbp_settings
        self.augment = augment
        self.use_lbp = lbp_settings is not None

        self.items = []  # (img_path, y, x, label)

        # caches
        self.image_cache = {}
        self.lbp_cache = {}  # {img_path: { (P,R): lbp_full }}

        for img_path, label in samples:
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            h, w = gray.shape

            # store RGB image
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            self.image_cache[img_path] = rgb
            
            # compute and store LBP(s)
            if self.use_lbp:
                self.lbp_cache[img_path] = {}
                for P, R in self.lbp_settings:
                    lbp = local_binary_pattern(
                        gray, P=P, R=R, method="uniform"
                    ).astype(np.float32)
                    lbp /= (P + 2)  # normalization
                    self.lbp_cache[img_path][(P,R)] = lbp
                    
            exclude_w = 309
            exclude_h = 122
            
            ex_x0 = w - exclude_w
            ex_y0 = h - exclude_h
            ex_x1 = w
            ex_y1 = h

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):

                    x2 = x + patch_size
                    y2 = y + patch_size

                    # safety (already ensured by range, but kept explicit)
                    if x2 > w or y2 > h:
                        continue

                    # -----------------------
                    # EXCLUDE bottom-right region
                    # check rectangle overlap
                    # -----------------------
                    overlap_x = not (x2 <= ex_x0 or x >= ex_x1)
                    overlap_y = not (y2 <= ex_y0 or y >= ex_y1)

                    if overlap_x and overlap_y:
                        continue  # skip patches touching excluded area

                    self.items.append((img_path, y, x, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, y, x, label = self.items[idx]

        # -----------------------
        # Extract RGB patch
        # -----------------------
        img = self.image_cache[img_path]  # (H, W, 3), float32 [0,1]
        rgb = img[y:y + self.patch_size, x:x + self.patch_size]

        # -----------------------
        # Extract LBP patches
        # -----------------------
        lbp_list = None
        if self.use_lbp:
            lbp_list = []
            for P, R in self.lbp_settings:
                lbp_full = self.lbp_cache[img_path][(P, R)]
                lbp_patch = lbp_full[y:y + self.patch_size, x:x + self.patch_size]
                lbp_list.append(lbp_patch)

        # -----------------------
        # Joint augmentation
        # -----------------------
        # Joint augmentation with copy to avoid negative strides
        if self.augment:
            do_hflip = np.random.rand() < 0.5
            do_vflip = np.random.rand() < 0.5

            if do_hflip:
                rgb = np.flip(rgb, axis=1).copy()
                if lbp_list is not None:
                    lbp_list = [np.flip(l, axis=1).copy() for l in lbp_list]

            if do_vflip:
                rgb = np.flip(rgb, axis=0).copy()
                if lbp_list is not None:
                    lbp_list = [np.flip(l, axis=0).copy() for l in lbp_list]

        # -----------------------
        # Convert to torch
        # -----------------------
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        if lbp_list is not None:
            lbp = torch.stack(
                [torch.from_numpy(l) for l in lbp_list],
                dim=0
            )  # (N_lbp, H, W)
            
        else:
            lbp = None

        return {
            "rgb": rgb,
            "lbp": lbp,
            "label": torch.tensor(label, dtype=torch.long),
            "img_path": str(img_path),
            "coords": torch.tensor(
                [x, y, x + self.patch_size, y + self.patch_size],
                dtype=torch.int32
            )
        }
