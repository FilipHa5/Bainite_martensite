import matplotlib.pyplot as plt

def plot_training_history(history, path_to_save=None):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # ----- Loss -----
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train")
    plt.plot(epochs_range, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ----- Accuracy -----
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train")
    plt.plot(epochs_range, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()

    # Save if path is provided
    if path_to_save:
        plt.savefig(path_to_save, bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()
    