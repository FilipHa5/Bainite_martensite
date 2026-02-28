import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

def train(model, optimizer, criterion, train_loader, device, num_epochs = 10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            images_rgb = batch["rgb"].to(device, non_blocking=True)
            images_lbp = (
                batch["lbp"].to(device, non_blocking=True)
                if batch["lbp"] is not None
                else None
            )
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images_rgb, images_lbp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_rgb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images_rgb.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.2f}%")

def train_single_val(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    patience=5,
    min_delta=0.0,
    plot=True,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # =========================
        # Train
        # =========================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            # print(batch["rgb"].shape, batch["lbp"].shape)
            images_rgb = batch["rgb"].to(device, non_blocking=True)
            images_lbp = batch["lbp"].to(device, non_blocking=True) if isinstance(batch["lbp"], torch.Tensor) else None
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images_rgb, images_lbp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_rgb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images_rgb.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # =========================
        # Validation
        # =========================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images_rgb = batch["rgb"].to(device, non_blocking=True)
                images_lbp = batch["lbp"].to(device, non_blocking=True) if batch["lbp"] is not None else None
                labels = batch["label"].to(device, non_blocking=True)

                outputs = model(images_rgb, images_lbp)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images_rgb.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images_rgb.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # =========================
        # Logging
        # =========================
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # =========================
        # Early stopping
        # =========================
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Restore best model
    model.load_state_dict(best_model_wts)

    # =========================
    # Plot
    # =========================
    if plot:
        epochs_range = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], label="Train")
        plt.plot(epochs_range, history["val_loss"], label="Val")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["train_acc"], label="Train")
        plt.plot(epochs_range, history["val_acc"], label="Val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model, history

def train_full(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    num_epochs=10,
):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images_rgb = batch["rgb"].to(device)
            images_lbp = batch["lbp"].to(device) if isinstance(batch["lbp"], torch.Tensor) else None
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images_rgb, images_lbp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_rgb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images_rgb.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    return model

def cross_validate(
    model_class,
    dataset,
    param_config,
    device,
    num_epochs=20,
    batch_size=32,
    n_splits=5
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Create NEW model each fold
        model = model_class(**param_config["model_params"]).to(device)

        optimizer = param_config["optimizer"](
            model.parameters(),
            **param_config["optimizer_params"]
        )

        criterion = param_config["criterion"]

        model, history = train_single_val(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            device,
            num_epochs=num_epochs,
            plot=False
        )

        best_val_acc = max(history["val_acc"])
        fold_scores.append(best_val_acc)

    mean_score = np.mean(fold_scores)
    print(f"\nMean CV Accuracy: {mean_score:.2f}%")

    return mean_score