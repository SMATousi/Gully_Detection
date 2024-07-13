import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from model import *
from dataset import *
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score



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

    if accelerator.is_main_process:
        if args.logging:
        
            wandb.init(
                    # set the wandb project where this run will be logged
                project=arg_projectname, name=arg_runname
                    
                    # track hyperparameters and run metadata
                    # config={
                    # "learning_rate": 0.02,
                    # "architecture": "CNN",
                    # "dataset": "CIFAR-100",
                    # "epochs": 20,
                    # }
            )
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = accelerator.device
    
    print(device)
    
    pos_dir = '/root/home/data/All_Pos_Neg/all_pos/rgb_images/'
    neg_dir = '/root/home/data/All_Pos_Neg/all_neg/rgb_images/'

    pos_dem_dir = '/root/home/data/All_Pos_Neg/all_pos/dem/'
    neg_dem_dir = '/root/home/data/All_Pos_Neg/all_neg/dem/'

    pos_gt_mask_dir = '/root/home/data/All_Pos_Neg/all_pos/ground_truth/'
    neg_gt_mask_dir = '/root/home/data/All_Pos_Neg/all_neg/ground_truth/'

    
    # dem_dir = '/root/home/data/dem'
    # so_dir = '/root/home/data/so'
    # rgb_dir = '/root/home/data/rgb'
    
    # pretrained_model_path = '/root/home/pre_trained/B3_rn50_moco_0099_ckpt.pth'
    
    #pretrained_model_path = '/home/macula/SMATousi/cluster/docker-images/dem2so_more_data/pre_models/B3_rn50_moco_0099_ckpt.pth'

    #pretrained_model_path = '/home/macula/SMATousi/cluster/docker-images/dem2so_more_data/pre_models/B3_rn50_moco_0099_ckpt.pth'

    
    
    batch_size = arg_batch_size
    learning_rate = 0.0001
    epochs = arg_epochs
    number_of_workers = 0
    image_size = arg_imagesize
    val_percent = 0.1
    
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    full_dataset = SixImageDataset_DEM_GT(pos_dir, 
                                 neg_dir, 
                                 pos_dem_dir,
                                 neg_dem_dir,
                                 pos_gt_mask_dir,
                                 neg_gt_mask_dir,
                                 transform=transform)
    # DataLoader
    
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train, val = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=arg_batch_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=arg_batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True, drop_last=True)
    
    
    
    print("Data is loaded")
    
    
    resnet_extractor = ResNetFeatureExtractor()
    mlp_classifier = MLPClassifier(input_size=6*2048, hidden_size=512, output_size=1)
    
    model = Gully_Classifier(input_size=6*2048, hidden_size=512, output_size=1)
    
    
    from torch.optim import Adam
    # criterion = nn.CrossEntropyLoss()
    # cldice_criterion = CE_CLDICE_Loss(alpha=arg_alpha, beta=arg_beta)
    # cldice_criterion = CE_CLDICE_Loss_optimized(alpha=arg_alpha, beta=arg_beta)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    validation_dataloader = accelerator.prepare(val_loader)
    
    # Training loop
    
    for epoch in range(epochs):

        # Training
        resnet_extractor.eval()  # Feature extractor should be in eval mode
        mlp_classifier.train()
        
        total_loss = 0
        all_labels = []
        all_preds = []
        
        
        for batch in tqdm(training_dataloader):
            images, dem_images, gt_masks, labels = batch
    
            # features = [resnet_extractor(image) for image in images]
            # stacked_features = torch.stack(features, dim=1)
            # output = mlp_classifier(stacked_features)
            output = model(images)
            loss = criterion(output.squeeze(), labels)

            all_predictions = accelerator.gather(output)
            all_targets = accelerator.gather(labels)

            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.round(all_predictions.squeeze()).detach().cpu().numpy()
            all_labels.extend(all_targets.cpu().numpy())
            all_preds.extend(preds)
    
            # Backward and optimize
            optimizer.zero_grad()
            
            optimizer.step()
            
            if arg_nottest:
                continue
            else:
                break
    
        # if arg_nottest:
        #     for k in train_metrics:
        #         train_metrics[k] /= len(training_dataloader)

        train_loss = total_loss / len(train_loader)
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        if accelerator.is_main_process:

            if args.logging:
                wandb.log({'Train/Loss':train_loss.item(),
                           'Train/Precision': train_precision,
                           'Train/Recall': train_recall,
                           'Train/F1': train_f1,
                           'Train/Epoch': epoch})
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item()}")
    
    
    
        # Validation loop
        resnet_extractor.eval()
        mlp_classifier.eval()
        
        total_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
    
            for batch in tqdm(validation_dataloader):
                images, dem_images, gt_masks, labels = batch


                # features = [resnet_extractor(image) for image in images]
                # stacked_features = torch.stack(features, dim=1)
                # output = mlp_classifier(stacked_features)
                output = model(images)
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
            
            # if arg_nottest:
            #     for k in val_metrics:
            #         val_metrics[k] /= len(validation_dataloader)

            val_loss = total_loss / len(val_loader)
            val_precision = precision_score(all_labels, all_preds)
            val_recall = recall_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds)
    
            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Validation/Loss':val_loss.item(),
                           'Validation/Precision': val_precision,
                           'Validation/Recall': val_recall,
                           'Validation/F1': val_f1,
                           'Validation/Epoch': epoch})
        
                    if (epoch + 1) % arg_savingstep == 0:
                        
                        os.makedirs('../saved_models', exist_ok=True)
                        torch.save(MLPClassifier.state_dict(), f'../saved_models/model_epoch_{epoch+1}.pth')
                        artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
                        artifact.add_file(f'../saved_models/model_epoch_{epoch+1}.pth')
                        wandb.log_artifact(artifact)
                        # save_comparison_figures(model, val_loader, epoch + 1, device)
                
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

