import torch
import os
from models import MicrostructureNet
# from utils import run_nested_cv
from save_params import StoreParams
from loaders import make_cv_loaders
from plot_training import plot_training_history
from torch import nn
from train import train_single_val
from nn_cv_trainer import run_nested_cv_dual_separate
data_root = os.path.join("images")

BATCH_SIZE=128
PATCH_SIZE=128
STRIDE=64
LBP_SETTINGS=None#[(16, 2), (24,3), (32, 4)]
LR = 1e-3
EPOCHS = 100

def main():
    # lbp_settings=[(16, 2), (24,3), (32, 4)]
    print("Hello from bainite-martensite!")
    # train_loader, val_loader, test_loader = make_train_val_test_loaders(
    #     data_root,
    #     batch_size=BATCH_SIZE,
    #     patch_size=256,
    #     stride=64,
    #     lbp_settings=LBP_SETTINGS,
    #     val_frac=0.15,
    #     test_frac=0.15,
    #     num_workers=0, # Windows Jupyter safe - multiprocessing issue
    #     random_state=42
    # )

    param_tracker = StoreParams()
    param_tracker.add("batch_size", BATCH_SIZE)
    param_tracker.add("patch_size", PATCH_SIZE)
    param_tracker.add("stride", STRIDE)
    param_tracker.add("LBP", LBP_SETTINGS)
    param_tracker.add("lr", LR)
    param_tracker.add("epochs", EPOCHS)
    
    result_path = param_tracker.save_params()

    # for fold, train_loader, val_loader in make_cv_loaders(
    #     data_root,
    #     batch_size=128,
    #     n_splits=4,
    #     patch_size=256,
    #     stride=64,
    #     lbp_settings=LBP_SETTINGS
    # ):
    
    

    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     print(f"Using device: {device}")

    #     for idx, (train_fold, val_fold) in enumerate(zip(train_loader, val_loader)):
    #         # ------------------ Model ------------------
    #         model = MicrostructureNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True).to(device)
    #         # --------------- Loss + Optimizer-----------
    #         criterion = nn.CrossEntropyLoss()
    #         optimizer = torch.optim.Adam(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=LR
    #         )

    #         print(f"Fold no:", idx)
    #         model, history = train_single_val(
    #             model,
    #             optimizer,
    #             criterion,
    #             train_loader,
    #             val_loader,
    #             device,
    #             num_epochs=EPOCHS,
    #             patience=7,
    #         )
            
    #         plot_training_history(history, result_path)
    
    def build_resnet_model():
        return MicrostructureNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True)
        
    def build_dt_model():
        raise NotImplementedError
    
    run_nested_cv_dual_separate(
        result_path,
        build_resnet_model,
        build_dt_model,
        data_root,
        outer_splits=4,
        inner_splits=3,
        batch_size=128,
        patch_size=256,
        stride=64,
        lbp_settings=LBP_SETTINGS,
        param_grid=None,
        device="cuda",
        epochs=EPOCHS
    )
    
        
    # missclassified, coords, probs, all_eval_paths, preds, trues, prob_vectors, report, cm = evaluate_and_visualize_single_head(
    #         model=model,
    #         loader=val_loader,
    #         device=device,
    #         max_show=len(val_loader.dataset)
    #     )
        
    #     param_tracker.add_classification_report("classification_report", report)

    #     perform_all_patches_corrections(coords, probs, all_eval_paths, preds, trues)

    #     # 1️⃣ Confidence heatmaps
    #     create_heatmaps_per_image(result_path,all_eval_paths, coords, probs, sigma=2)

    #     # 2️⃣ Misclassified patches
    #     plot_misclassified(result_path, missclassified, dt_classifier)

    #     # 3️⃣ Misclassification density
    #     create_misclassification_density_maps(result_path,all_eval_paths, coords, preds, trues)

    #     # 4️⃣ Per-class error maps (example target_class = 0)
    #     create_per_class_error_maps(result_path, all_eval_paths, coords, preds, trues, target_class=0)

    #     # 5️⃣ Uncertainty maps
    #     create_uncertainty_maps(result_path, all_eval_paths, coords, prob_vectors)

        
    # # ------------------ Model ------------------
    # model = MicrostructureNet(lbp_settings=LBP_SETTINGS, freeze_backbone=True).to(device)
    # # --------------- Loss + Optimizer-----------
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=LR
    # )
    
    # # for fold, train_loader, val_loader in make_cv_loaders
    # dt_classifier = train_dt_model(train_loader, val_loader)

    # model, history = train_single_val(
    #     model,
    #     optimizer,
    #     criterion,
    #     train_loader,
    #     val_loader,
    #     device,
    #     num_epochs=EPOCHS,
    #     patience=7,
    # )
    
    
    # param_tracker.add("epochs", 
    #                   len(history))

    # missclassified, coords, probs, all_eval_paths, preds, trues, prob_vectors, report, cm = evaluate_and_visualize_single_head(
    #     model=model,
    #     loader=val_loader,
    #     device=device,
    #     max_show=len(val_loader.dataset)
    # )
    
    # param_tracker.add_classification_report("classification_report", report)

    # perform_all_patches_corrections(coords, probs, all_eval_paths, preds, trues)

    # # 1️⃣ Confidence heatmaps
    # create_heatmaps_per_image(result_path,all_eval_paths, coords, probs, sigma=2)

    # # 2️⃣ Misclassified patches
    # plot_misclassified(result_path, missclassified, dt_classifier)

    # # 3️⃣ Misclassification density
    # create_misclassification_density_maps(result_path,all_eval_paths, coords, preds, trues)

    # # 4️⃣ Per-class error maps (example target_class = 0)
    # create_per_class_error_maps(result_path, all_eval_paths, coords, preds, trues, target_class=0)

    # # 5️⃣ Uncertainty maps
    # create_uncertainty_maps(result_path, all_eval_paths, coords, prob_vectors)

if __name__ == "__main__":
    main()
