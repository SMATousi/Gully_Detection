from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
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
    parser.add_argument("--pret_epochs", type=int, default=70)
    parser.add_argument("--lp_epochs", type=int, default=30)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--dropoutrate", type=float, default=0.5)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")
    parser.add_argument("--logging", help="Enable verbose mode", action="store_true")
    parser.add_argument("--init_embed", help="Enable verbose mode", action="store_true")
    parser.add_argument("--freeze_weight", help="Enable verbose mode", action="store_true")
    
    args = parser.parse_args()
    
    arg_batch_size = args.batchsize
    # arg_epochs = args.epochs
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

    if args.init_embed:
        init_embed = True 
    else:
        init_embed = False

    if args.freeze_weight:
        freeze_weight = True 
    else:
        freeze_weight = False
    
    
    # args = parser.parse_args()

    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = accelerator.device
    
    print(device)

    # Load dataset and initialize model, criterion, optimizer
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    pos_dir = '/root/home/data/Pos_Neg_224/pos/rgb_images/'
    neg_dir = '/root/home/data/Pos_Neg_224/neg/rgb_images/'
    pos_dem_dir = '/root/home/data/Pos_Neg_224/pos/dem/'
    neg_dem_dir = '/root/home/data/Pos_Neg_224/neg/dem/'
    pos_gt_mask_dir = '/root/home/data/Pos_Neg_224/pos/ground_truth/'
    neg_gt_mask_dir = '/root/home/data/Pos_Neg_224/neg/ground_truth/'

    full_dataset = SixImageDataset_DEM_GT(
        pos_dir, neg_dir, pos_dem_dir, neg_dem_dir, pos_gt_mask_dir, neg_gt_mask_dir,
        transform=transform
    )


    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
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

        # train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        np.random.shuffle(train_idx)

        split_point = int(0.75 * len(train_idx))
        train_subset_pretraining_idx = train_idx[:split_point]  # 75% for task 1
        train_subset_lp_idx = train_idx[split_point:]  # 25% for task 2

        train_subset_pretraining = torch.utils.data.Subset(full_dataset, train_subset_pretraining_idx)
        train_subset_lp = torch.utils.data.Subset(full_dataset, train_subset_lp_idx)

        train_loader_pretraining = DataLoader(train_subset_pretraining, batch_size=args.batchsize, shuffle=True, num_workers=8)
        train_loader_lp = DataLoader(train_subset_lp, batch_size=args.batchsize, shuffle=True, num_workers=8)

        # Assuming val_loader remains unchanged
        val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=8)

        # train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        # val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=8)

        # Initialize resnet_extractor and mlp_classifier
        # resnet_extractor = ResNetFeatureExtractor()
        # resnet_extractor.eval()  # Ensure resnet_extractor is always in eval mode
        # mlp_classifier = MLPClassifier(input_size=6*2048, hidden_size=512, output_size=1)

        # Wrap mlp_classifier with accelerator
        # mlp_classifier = accelerator.prepare(mlp_classifier)

        model = ViT_Gully_Classifier(tandom_init_embeddings=init_embed)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        model, optimizer, training_dataloader_pretraining, scheduler = accelerator.prepare(
        model, optimizer, train_loader_pretraining, scheduler
        )

        training_dataloader_lp = accelerator.prepare(train_loader_lp)
        validation_dataloader = accelerator.prepare(val_loader)

        train_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
        val_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}

################################################################################################
################################################################################################
####################### Pretraining Starts Here ################################################
################################################################################################
################################################################################################

        for epoch in range(args.pret_epochs):
            # Training
            # resnet_extractor.eval()  # Feature extractor should be in eval mode
            model.train()
            
            total_loss = 0
            all_labels = []
            all_preds = []

            for batch in tqdm(training_dataloader_pretraining):
                images, dem_images, gt_masks, labels = batch
                # print(images[0].shape)

                output = model(images)
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

            if accelerator.is_main_process:

                if args.logging:
                    wandb.log({'Train/Loss':train_loss,
                            'Train/Precision': train_precision,
                            'Train/Recall': train_recall,
                            'Train/F1': train_f1,
                            'Train/Epoch': epoch})

            train_metrics['loss'].append(train_loss)
            train_metrics['precision'].append(train_precision)
            train_metrics['recall'].append(train_recall)
            train_metrics['f1'].append(train_f1)

            # Validation Loop
            # mlp_classifier.eval()
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

            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Validation/Loss':val_loss,
                           'Validation/Precision': val_precision,
                           'Validation/Recall': val_recall,
                           'Validation/F1': val_f1,
                           'Validation/Epoch': epoch})

            val_metrics['loss'].append(val_loss)
            val_metrics['precision'].append(val_precision)
            val_metrics['recall'].append(val_recall)
            val_metrics['f1'].append(val_f1)

            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss}, Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}")
            print(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {val_loss}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1: {val_f1}")



################################################################################################
################################################################################################
####################### Pretraining Ends Here ##################################################
################################################################################################
################################################################################################

        for epoch in range(args.lp_epochs):
            # Training
            # resnet_extractor.eval()  # Feature extractor should be in eval mode
            model.train()
            
            total_loss = 0
            all_labels = []
            all_preds = []

            for batch in tqdm(training_dataloader_lp):
                images, dem_images, gt_masks, labels = batch
                # print(images[0].shape)

                output = model(images,freeze_the_params=freeze_weight)
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

            if accelerator.is_main_process:

                if args.logging:
                    wandb.log({'Train/Loss':train_loss,
                            'Train/Precision': train_precision,
                            'Train/Recall': train_recall,
                            'Train/F1': train_f1,
                            'Train/Epoch': epoch})

            train_metrics['loss'].append(train_loss)
            train_metrics['precision'].append(train_precision)
            train_metrics['recall'].append(train_recall)
            train_metrics['f1'].append(train_f1)

            # Validation Loop
            # mlp_classifier.eval()
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

            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Validation/Loss':val_loss,
                           'Validation/Precision': val_precision,
                           'Validation/Recall': val_recall,
                           'Validation/F1': val_f1,
                           'Validation/Epoch': epoch})

            val_metrics['loss'].append(val_loss)
            val_metrics['precision'].append(val_precision)
            val_metrics['recall'].append(val_recall)
            val_metrics['f1'].append(val_f1)

            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss}, Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}")
            print(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {val_loss}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1: {val_f1}")

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

if __name__ == "__main__":
    main()
