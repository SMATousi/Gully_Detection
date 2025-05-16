from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset import SixImageDataset_DEM_GT
from model import ResNetFeatureExtractor, MLPClassifier
import argparse
from model import *
from dataset import *
from utils import *
from timm_flexiViT import Flexi_ViT_Gully_Classifier
from tqdm import tqdm

# random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    


    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    current_rank = accelerator.state.process_index
    num_gpus = accelerator.state.num_processes
    
    
    
    parser = argparse.ArgumentParser(description="A script with argparse options")
    
    # Add an argument for an integer option
    parser.add_argument("--runname", type=str, required=False)
    parser.add_argument("--projectname", type=str, required=False)
    parser.add_argument("--modelname", type=str, required=False)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--savingstep", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--dropoutrate", type=float, default=0.5)
    parser.add_argument("--gt_choice", type=str, required=True)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")
    parser.add_argument("--logging", help="Enable verbose mode", action="store_true")
    
    args = parser.parse_args()
    
    arg_batch_size = args.batchsize
    arg_epochs = args.epochs
    arg_runname = args.runname
    arg_projectname = args.projectname
    arg_modelname = args.modelname
    arg_savingstep = args.savingstep
    arg_threshold = args.threshold
    arg_imagesize = args.imagesize
    arg_dropoutrate = args.dropoutrate
    arg_alpha = args.alpha
    arg_beta = args.beta

    
    if args.nottest:
        arg_nottest = True 
    else:
        arg_nottest = False
    
    
    args = parser.parse_args()

    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = accelerator.device
    
    print(device)

    # Load dataset and initialize model, criterion, optimizer
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor()
    # ])

    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(20, expand=False),
        transforms.ToTensor()
    ])

    pos_dir = '/root/home/data/pos/rgb_images'  # Directory with positive tile images
    neg_dir = '/root/home/data/neg/rgb_images'  # Directory with negative tile images
    pos_dem_dir = '/root/home/data/pos/dem' # Directory with positive DEM tiles
    neg_dem_dir = '/root/home/data/neg/dem' # Directory with negative DEM tiles
    pos_gt_mask_dir = '/root/home/data/pos/ground_truth' # Directory with positive GT masks
    neg_gt_mask_dir = '/root/home/data/neg/ground_truth' # Directory with negative GT masks

    if args.gt_choice == 'neg_strict':
        json_path = '../labeling_tool/v2_results/final/agg/neg_strict_labels_majority.json'
    elif args.gt_choice == 'neg_len':
        json_path = '../labeling_tool/v2_results/final/agg/neg_len_1or0_labels_majority.json'
    elif args.gt_choice == 'pos_strict':
        json_path = '../labeling_tool/v2_results/final/agg/pos_strict_labels_majority.json'
    elif args.gt_choice == 'pos_len':
        json_path = '../labeling_tool/v2_results/final/agg/pos_len_4-1_labels_majority.json'
    else:
        raise ValueError(f"Invalid gt_choice: {args.gt_choice}")
    
    print("Using json path:", json_path)
    
    full_dataset = EightImageDataset_DEM_GT_Geo_from_JSON(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        pos_dem_dir=pos_dem_dir,
        neg_dem_dir=neg_dem_dir,
        pos_gt_mask_dir=pos_gt_mask_dir,
        neg_gt_mask_dir=neg_gt_mask_dir,
        labels_json_path=json_path,
        transform=augmentation,
        oversample=False
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"Starting Fold {fold + 1}")

        if accelerator.is_main_process:
            if args.logging:
            
                wandb.init(
                        # set the wandb project where this run will be logged
                    project=arg_projectname, name=f"{arg_runname}_{fold+1}"
                        
                        # track hyperparameters and run metadata
                        # config={
                        # "learning_rate": 0.02,
                        # "architecture": "CNN",
                        # "dataset": "CIFAR-100",
                        # "epochs": 20,
                        # }
                )

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=16)

        print("Train dataset size:", len(train_loader.dataset))
        print("Validation dataset size:", len(val_loader.dataset))

        # Initialize resnet_extractor and mlp_classifier
        # resnet_extractor = ResNetFeatureExtractor()
        # resnet_extractor.eval()  # Ensure resnet_extractor is always in eval mode
        # mlp_classifier = MLPClassifier(input_size=6*2048, hidden_size=512, output_size=1)

        # # Wrap mlp_classifier with accelerator
        # mlp_classifier = accelerator.prepare(mlp_classifier)

        model = Flexi_ViT_Gully_Classifier()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
        )
        validation_dataloader = accelerator.prepare(val_loader)

        train_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}

        for epoch in range(args.epochs):
            # Training
            model.train()
            
            total_loss = 0
            all_labels = []
            all_preds = []

            for batch in tqdm(training_dataloader):
                images, dem_images, gt_masks, labels = batch

                # print("labels", labels)

                output = model(images)
                # print("output", output.shape)
                loss = criterion(output.squeeze(), labels)

                total_loss += loss.item()

                all_predictions = accelerator.gather(output)
                all_targets = accelerator.gather(labels)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                preds = torch.round(all_predictions.squeeze()).detach().cpu().numpy()
                all_labels.extend(all_targets.cpu().numpy())
                all_preds.extend(preds)

                if arg_nottest:
                    continue
                else:
                    break

            train_loss = total_loss / len(train_loader)
            train_precision = precision_score(all_labels, all_preds)
            train_recall = recall_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)

            # Calculate confusion matrix and derived metrics
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
            train_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            train_for = fn / (fn + tn) if (fn + tn) > 0 else 0.0

            if accelerator.is_main_process:

                if args.logging:
                    wandb.log({'Train/Loss':train_loss,
                            'Train/Precision': train_precision,
                            'Train/Recall': train_recall,
                            'Train/F1': train_f1,
                            'Train/NPV': train_npv,
                            'Train/FOR': train_for,
                            'Train/Epoch': epoch})

            train_metrics['loss'].append(train_loss)
            train_metrics['precision'].append(train_precision)
            train_metrics['recall'].append(train_recall)
            train_metrics['f1'].append(train_f1)
            train_metrics.setdefault('npv', []).append(train_npv)
            train_metrics.setdefault('for', []).append(train_for)

            # Validation Loop
            model.eval()

            total_loss = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for batch in tqdm(validation_dataloader):
                    images, dem_images, gt_masks, labels = batch

                    # Extract features using resnet_extractor
                    output = model(images)

                    # Forward pass through mlp_classifier
                    loss = criterion(output.squeeze(), labels)

                    all_predictions = accelerator.gather(output)
                    all_targets = accelerator.gather(labels)
        
                    total_loss += loss.item()

                    preds = torch.round(all_predictions.squeeze()).detach().cpu().numpy()
                    all_labels.extend(all_targets.cpu().numpy())
                    all_preds.extend(preds)

                    if arg_nottest:
                            continue
                    else:
                        break

            val_loss = total_loss / len(val_loader)
            val_precision = precision_score(all_labels, all_preds)
            val_recall = recall_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds)

            # Calculate confusion matrix and derived metrics
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
            val_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            val_for = fn / (fn + tn) if (fn + tn) > 0 else 0.0

            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Validation/Loss':val_loss,
                           'Validation/Precision': val_precision,
                           'Validation/Recall': val_recall,
                           'Validation/F1': val_f1,
                           'Validation/NPV': val_npv,
                           'Validation/FOR': val_for,
                           'Validation/Epoch': epoch})
                    
                    if (epoch + 1) % arg_savingstep == 0:
                        
                        os.makedirs('../saved_models', exist_ok=True)
                        torch.save(model.state_dict(), f'../saved_models/model_epoch_{epoch+1}.pth')
                        artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
                        artifact.add_file(f'../saved_models/model_epoch_{epoch+1}.pth')
                        wandb.log_artifact(artifact)
                        # save_comparison_figures(model, val_loader, epoch + 1, device)
                

            val_metrics['loss'].append(val_loss)
            val_metrics['precision'].append(val_precision)
            val_metrics['recall'].append(val_recall)
            val_metrics['f1'].append(val_f1)
            val_metrics.setdefault('npv', []).append(val_npv)
            val_metrics.setdefault('for', []).append(val_for)

            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss}, Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}, Train NPV: {train_npv}, Train FOR: {train_for}")
            print(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {val_loss}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1: {val_f1}, Val NPV: {val_npv}, Val FOR: {val_for}")

        fold_metrics.append((train_metrics, val_metrics))

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:

            if args.logging:
                
                
                wandb.finish()
    
    # Calculate average metrics
    avg_train_metrics = {key: np.mean([np.mean(fold[0][key]) for fold in fold_metrics]) for key in fold_metrics[0][0]}
    avg_val_metrics = {key: np.mean([np.mean(fold[1][key]) for fold in fold_metrics]) for key in fold_metrics[0][1]}

    print("Average Train Metrics:", avg_train_metrics)
    print("Average Validation Metrics:", avg_val_metrics)
    if 'npv' in avg_train_metrics and 'for' in avg_train_metrics:
        print(f"Average Train NPV: {avg_train_metrics['npv']}, Average Train FOR: {avg_train_metrics['for']}")
    if 'npv' in avg_val_metrics and 'for' in avg_val_metrics:
        print(f"Average Validation NPV: {avg_val_metrics['npv']}, Average Validation FOR: {avg_val_metrics['for']}")

    

if __name__ == "__main__":
    main()
