import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

from sklearn.model_selection import GridSearchCV

def normalized_histogram(img, bins=128):
    if not isinstance(bins, int):
        raise TypeError(f"bins must be int, got {type(bins)}")

    if bins <= 0:
        raise ValueError("bins must be > 0")
    
    # Flatten image to 1D
    pixels = img.ravel()
    
    hist, bin_edges = np.histogram(
        pixels,
        bins=bins,
        range=(0, 1),
        density=True
    )
       
    return hist

# def loader_to_numpy(loader, device="cpu", bins=128):
#     """
#     Converts a PyTorch DataLoader into numpy arrays for training non-PyTorch models.
#     Safely handles missing LBP by replacing with zeros.
    
#     Args:
#         loader: PyTorch DataLoader yielding dicts with keys "rgb", "lbp", "label"
#         device: device to move tensors to (usually "cpu")
#         bins: number of histogram bins for RGB/LBP features
    
#     Returns:
#         X: numpy array of features [num_samples, bins_rgb + bins_lbp]
#         y: numpy array of labels
#     """
#     import numpy as np

#     X = []
#     y = []

#     for item in loader:
#         # Move to CPU
#         rgb_batch = item["rgb"].to("cpu").numpy()
#         label_batch = item["label"].to("cpu").numpy()

#         # If LBP exists, convert to numpy, else use zeros
#         if "lbp" in item and item["lbp"] is not None:
#             lbp_batch = item["lbp"].to("cpu").numpy()
#         else:
#             lbp_batch = np.zeros_like(rgb_batch)  # shape matches RGB for safety

#         # Iterate batch samples
#         for idx in range(len(rgb_batch)):
#             rgb_hist = normalized_histogram(rgb_batch[idx], bins)

#             if lbp_batch is not None:
#                 lbp_hist = normalized_histogram(lbp_batch[idx], bins)
#             else:
#                 lbp_hist = np.zeros(bins, dtype=np.float32)

#             features = np.concatenate([rgb_hist, lbp_hist])
#             X.append(features)
#             y.append(label_batch[idx])

#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.int64)

#     print("Final X shape:", X.shape)
#     print("Final y shape:", y.shape)

#     return X, y

def loader_to_numpy(loader, device, bins=128):
    X = []
    y = []

    for item in loader:
        rgb_batch = item["rgb"].cpu().numpy()
        label_batch = item["label"].cpu().numpy()

        # Check if LBP exists in this batch
        has_lbp = "lbp" in item and item["lbp"] is not None

        if has_lbp:
            lbp_batch = item["lbp"].cpu().numpy()
        else:
            lbp_batch = [None] * len(rgb_batch)

        for img, lbp, label in zip(rgb_batch, lbp_batch, label_batch):

            rgb_hist = normalized_histogram(img, bins)

            if lbp is None:
                lbp_hist = np.zeros(bins, dtype=np.float32)
            else:
                lbp_hist = normalized_histogram(lbp, bins)

            features = np.concatenate([rgb_hist, lbp_hist])
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # print("Final X shape:", X.shape)

    return X, y


def build_dt_model():
    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    )

    return clf


def train_dt_model(train_loader, val_loader, bins=128):
    device = "cpu"
    X_train, y_train = loader_to_numpy(train_loader, device, bins)
    X_val, y_val = loader_to_numpy(val_loader, device, bins)

    clf = build_dt_model()

    clf.fit(X_train, y_train)
    clf.bins_used = bins
    
    # Predict on validation set
    y_pred = clf.predict(X_val)

    # Compute accuracy
    val_acc = accuracy_score(y_val, y_pred)

    return clf, val_acc


def predict_on_missclassified(clf, loader, device):
    X, y = loader_to_numpy(loader, device, clf.bins_used)
    print("Prediction X shape:", X.shape)
    print("Model expects:", clf.n_features_in_)
    preds = clf.predict(X)
    return preds, y