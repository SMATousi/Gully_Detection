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
from model import Gully_Classifier
import argparse
from tqdm import tqdm

# random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define a function to train and evaluate on a fold
def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, accelerator, epochs):
    train_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    val_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    for epoch in range(epochs):
        model.resnet_extractor.eval()  # Feature extractor should be in eval mode
        model.mlp_classifier.train()
        
        total_loss = 0
        all_labels = []
        all_preds = []

        # Training Loop
        model.train()
        for batch in tqdm(train_loader):
            images, dem_images, gt_masks, labels = batch
            output = model(images)
            loss = criterion(output.squeeze(), labels)

            total_loss += loss.item()

            all_predictions = accelerator.gather(output)
            all_targets = accelerator.gather(labels)

            preds = torch.round(all_predictions.squeeze()).detach().cpu().numpy()
            all_labels.extend(all_targets.cpu().numpy())
            all_preds.extend(preds)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if arg_nottest:
                continue
            else:
                break


            # preds = torch.round(output.squeeze()).detach().cpu().numpy()
            # labels_np = labels.cpu().numpy()

        train_metrics['loss'] = total_loss / len(train_loader)
        train_metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
        train_metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
        train_metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)

        if accelerator.is_main_process:

            if args.logging:
                wandb.log({'Train/Loss':train_metrics['loss'],
                           'Train/Precision': train_metrics['precision'],
                           'Train/Recall': train_metrics['recall'],
                           'Train/F1': train_metrics['f1'],
                           'Train/Epoch': epoch})

        # Validation Loop
        model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, dem_images, gt_masks, labels = batch
                output = model(images)
                loss = criterion(output.squeeze(), labels)
                total_loss += loss.item()

                all_predictions = accelerator.gather(output)
                all_targets = accelerator.gather(labels)
                preds = torch.round(all_predictions.squeeze()).detach().cpu().numpy()
                all_labels.extend(all_targets.cpu().numpy())
                all_preds.extend(preds)

                if arg_nottest:
                    continue
                else:
                    break

                # preds = torch.round(output.squeeze()).detach().cpu().numpy()
                # labels_np = labels.cpu().numpy()

            val_metrics['loss'] = total_loss / len(val_loader)
            val_metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
            val_metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
            val_metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)

            if accelerator.is_main_process:

            if args.logging:
                wandb.log({'Train/Loss':val_metrics['loss'],
                           'Train/Precision': val_metrics['precision'],
                           'Train/Recall': val_metrics['recall'],
                           'Train/F1': val_metrics['f1'],
                           'Train/Epoch': epoch})

        # for key in val_metrics:
        #     val_metrics[key] /= len(val_loader)

    return train_metrics, val_metrics


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

    

    device = accelerator.device
    
    print(device)

    # Load dataset and initialize model, criterion, optimizer
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    pos_dir = '/root/home/data/All_Pos_Neg/all_pos/rgb_images/'
    neg_dir = '/root/home/data/All_Pos_Neg/all_neg/rgb_images/'
    pos_dem_dir = '/root/home/data/All_Pos_Neg/all_pos/dem/'
    neg_dem_dir = '/root/home/data/All_Pos_Neg/all_neg/dem/'
    pos_gt_mask_dir = '/root/home/data/All_Pos_Neg/all_pos/ground_truth/'
    neg_gt_mask_dir = '/root/home/data/All_Pos_Neg/all_neg/ground_truth/'

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

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False)

        model = Gully_Classifier(input_size=6*2048, hidden_size=512, output_size=1)
        # model = accelerator.prepare(model)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        
        model, optimizer, training_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        validation_dataloader = accelerator.prepare(val_loader)

        train_metrics, val_metrics = train_and_evaluate(training_dataloader, 
                                                        validation_dataloader, 
                                                        model, 
                                                        criterion, 
                                                        optimizer, 
                                                        accelerator, 
                                                        arg_epochs)

        fold_metrics.append((train_metrics, val_metrics))

        print(f"Fold {fold + 1} - Train Metrics: {train_metrics}, Validation Metrics: {val_metrics}")

    # Calculate average metrics
    avg_train_metrics = {key: np.mean([fold[0][key] for fold in fold_metrics]) for key in fold_metrics[0][0]}
    avg_val_metrics = {key: np.mean([fold[1][key] for fold in fold_metrics]) for key in fold_metrics[0][1]}

    print("Average Train Metrics:", avg_train_metrics)
    print("Average Validation Metrics:", avg_val_metrics)

if __name__ == "__main__":
    main()
