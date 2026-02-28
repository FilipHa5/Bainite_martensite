import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

from utils import make_cv_loaders, make_train_val_test_loaders
from utils import StoreParams
from sklearn.model_selection import GridSearchCV

BATCH_SIZE=128
PATCH_SIZE=128
STRIDE=64
LBP_SETTINGS=[(16,2)]
LR = 1e-3
EPOCHS = 50

data_root = os.path.join("images")


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

import numpy as np

def normalized_histogram(img, bins=128):
    # Flatten image to 1D
    pixels = img.ravel()
    
    hist, bin_edges = np.histogram(
        pixels,
        bins=bins,
        range=(0, 1),
        density=True
    )
       
    return hist
    
def loader_to_numpy(loader, device):
    X_rgb = []
    X_lbp = []
    y = []

            # "rgb": rgb,
            # "lbp": lbp,
            # "label": torch.tensor(label, dtype=torch.long),
            # "img_path": str(img_path),
            # "coords": torch.tensor(
            #     [x, y, x + self.patch_size, y + self.patch_size],
            #     dtype=torch.int32
            # )
    for item in loader: 
        rgb_batch = item["rgb"].cpu().numpy()
        lbp_batch = item["lbp"].cpu().numpy()
        label_batch = item["label"].cpu().numpy()
        for img, lbp, label in zip(rgb_batch, lbp_batch, label_batch):
            
            X_rgb.append(normalized_histogram(img))
            X_lbp.append(normalized_histogram(lbp))
            y.append(label)
            
    X_rgb = np.array(X_rgb)
    X_lbp = np.array(X_lbp)
    X = [np.hstack((rgb, lbp)) for rgb, lbp in zip(X_rgb, X_lbp)]
    X = np.array(X)

    y = np.hstack(y)

    return X, y

if __name__ == "__main__":
    device = "cpu"
    X_train, y_train = loader_to_numpy(train_loader, device)
    X_val, y_val = loader_to_numpy(val_loader, device)
    X_test, y_test = loader_to_numpy(test_loader, device)

    print(X_train.shape, X_test.shape)
    print(X_train[0])

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': [None, 'sqrt', 'log2']
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    # clf = DecisionTreeClassifier(
    #     max_depth=10,
    #     random_state=42,
    #     min_samples_leaf=5,
    #     class_weight='balanced'
    # )

    # Fit the model
    # clf.fit(X_train, y_train)
    clf = grid.best_estimator_

    print("Best parameters:", grid.best_params_)

    # Validation predictions
    val_preds = clf.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_preds))
    print("Validation Classification Report:\n", classification_report(y_val, val_preds))

    # Test predictions
    test_preds = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print("Test Classification Report:\n", classification_report(y_test, test_preds))