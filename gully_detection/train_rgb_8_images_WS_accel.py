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
from timm_flexiViT import Flexi_ViT_Gully_Classifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix



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
    parser.add_argument("--numworkers", type=int, default=0)
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
    arg_numworkers = args.numworkers
    
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
    
    train_images_dir = '/root/home/pvc/large_unlabeled_dataset/rgb_images/'
    test_images_dir = '/root/home/pvc/final_pos_neg_test_data_merging_25_2/all_tiles/'
    train_label_model_results_dir = '../weak-supervision/large_dataset/results/train_lable_model_results.json'
    test_GT_labels_dir = '../labeling_tool/v2_results/final/agg/neg_strict_labels_majority.json'

    
    batch_size = arg_batch_size
    learning_rate = 0.0001
    epochs = arg_epochs
    number_of_workers = arg_numworkers
    image_size = arg_imagesize
    val_percent = 0.1
    
    
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    train_dataset = EightImageDataset_WS(train_images_dir,
                                 train_label_model_results_dir,
                                 transform=transform)
    # DataLoader
    
    test_dataset = EightImageDataset_WS_GT(test_images_dir,
                                 test_GT_labels_dir,
                                 transform=transform)
    

    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train, val = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=arg_batch_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=arg_batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=arg_batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True, drop_last=True)
    
    
    
    print("Data is loaded")
    
    
    model = Flexi_ViT_Gully_Classifier()

    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
    )
    validation_dataloader = accelerator.prepare(val_loader)
    test_dataloader = accelerator.prepare(test_loader)

    train_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
    test_metrics = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
    
    
    from torch.optim import Adam
    # criterion = nn.CrossEntropyLoss()
    # cldice_criterion = CE_CLDICE_Loss(alpha=arg_alpha, beta=arg_beta)
    # cldice_criterion = CE_CLDICE_Loss_optimized(alpha=arg_alpha, beta=arg_beta)
    
    # Training loop
    
    for epoch in range(epochs):

        model.train()
            
        total_loss = 0
        all_labels = []
        all_preds = []
        
        
        for batch in tqdm(training_dataloader):
            images, target_probs, target_label = batch
    
            # features = [resnet_extractor(image) for image in images]
            # stacked_features = torch.stack(features, dim=1)
            # output = mlp_classifier(stacked_features)
            output = model(images)

            loss = criterion(F.log_softmax(output, dim=1), target_probs)

            predicted_labels = torch.argmax(output, dim=1)

            all_predictions = accelerator.gather(predicted_labels)
            all_targets = accelerator.gather(target_label)

            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = all_predictions.cpu().numpy()
            all_labels.extend(all_targets.cpu().numpy())
            all_preds.extend(preds)
    
            # Backward and optimize
            optimizer.zero_grad()
            optimizer.step()
            
            if arg_nottest:
                continue
            else:
                break
        scheduler.step()
        train_loss = total_loss / len(training_dataloader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        # Confusion matrix for additional metrics
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        train_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        train_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        train_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        train_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        if accelerator.is_main_process:

            if args.logging:
                    wandb.log({'Train/Loss': train_loss,
                               'Train/Accuracy': train_accuracy,
                               'Train/Precision': train_precision,
                               'Train/Recall': train_recall,
                               'Train/F1': train_f1,
                               'Train/Specificity': train_specificity,
                               'Train/NPV': train_npv,
                               'Train/FPR': train_fpr,
                               'Train/FNR': train_fnr,
                               'Train/TP': tp,
                               'Train/TN': tn,
                               'Train/FP': fp,
                               'Train/FN': fn,
                               'Train/Epoch': epoch})
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item()}")
    
    
    
        # Validation loop
        model.eval()
        
        total_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
    
            for batch in tqdm(validation_dataloader):
                images, target_probs, target_label = batch


                # features = [resnet_extractor(image) for image in images]
                # stacked_features = torch.stack(features, dim=1)
                # output = mlp_classifier(stacked_features)
                output = model(images)
                predicted_labels = torch.argmax(F.log_softmax(output, dim=1), dim=1)
                loss = criterion(F.log_softmax(output, dim=1), target_probs)

                all_predictions = accelerator.gather(predicted_labels)
                all_targets = accelerator.gather(target_label)
    
                total_loss += loss.item()

                preds = all_predictions.cpu().numpy()
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
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, zero_division=0)
            val_recall = recall_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds)

            # Confusion matrix for additional metrics
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            val_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            val_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Validation/Loss': val_loss,
                               'Validation/Accuracy': val_accuracy,
                               'Validation/Precision': val_precision,
                               'Validation/Recall': val_recall,
                               'Validation/F1': val_f1,
                               'Validation/Specificity': val_specificity,
                               'Validation/NPV': val_npv,
                               'Validation/FPR': val_fpr,
                               'Validation/FNR': val_fnr,
                               'Validation/TP': tp,
                               'Validation/TN': tn,
                               'Validation/FP': fp,
                               'Validation/FN': fn,
                               'Validation/Epoch': epoch})
        
                    if (epoch + 1) % arg_savingstep == 0:
                        
                        os.makedirs('../saved_models', exist_ok=True)
                        torch.save(model.state_dict(), f'../saved_models/model_epoch_{epoch+1}.pth')
                        artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
                        artifact.add_file(f'../saved_models/model_epoch_{epoch+1}.pth')
                        wandb.log_artifact(artifact)
                        # save_comparison_figures(model, val_loader, epoch + 1, device)
        
        model.eval()
        
        # total_loss = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
    
            for batch in tqdm(test_dataloader):
                images, labels = batch


                # features = [resnet_extractor(image) for image in images]
                # stacked_features = torch.stack(features, dim=1)
                # output = mlp_classifier(stacked_features)
                output = model(images)
                predicted_labels = torch.argmax(F.log_softmax(output, dim=1), dim=1)
                # loss = criterion(F.log_softmax(output, dim=1), labels)

                all_predictions = accelerator.gather(predicted_labels)
                all_targets = accelerator.gather(labels)
    
                # total_loss += loss.item()

                preds = all_predictions.cpu().numpy()
                all_labels.extend(all_targets.cpu().numpy())
                all_preds.extend(preds)

                if arg_nottest:
                        continue
                else:
                    break
            
            # if arg_nottest:
            #     for k in val_metrics:
            #         val_metrics[k] /= len(validation_dataloader)

            # _loss = total_loss / len(val_loader)
            test_accuracy = accuracy_score(all_labels, all_preds)
            test_precision = precision_score(all_labels, all_preds, zero_division=0)
            test_recall = recall_score(all_labels, all_preds)
            test_f1 = f1_score(all_labels, all_preds)

            # Confusion matrix for additional metrics
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
            if accelerator.is_main_process:

                if args.logging:

                    wandb.log({'Test/Accuracy': test_accuracy,
                               'Test/Precision': test_precision,
                               'Test/Recall': test_recall,
                               'Test/F1': test_f1,
                               'Test/Specificity': test_specificity,
                               'Test/NPV': test_npv,
                               'Test/FPR': test_fpr,
                               'Test/FNR': test_fnr,
                               'Test/TP': tp,
                               'Test/TN': tn,
                               'Test/FP': fp,
                               'Test/FN': fn,
                               'Test/Epoch': epoch})
        
                
                

                
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

