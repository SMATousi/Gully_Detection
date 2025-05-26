#!/usr/bin/env python3
import os
import json
import glob
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def extract_tile_number(filename):
    """Extract tile number from a filename like 'rgb_0_tile_123.tif'."""
    match = re.search(r'tile_(\d+)', filename)
    if match:
        return match.group(1)
    return None

def build_ground_truth_dict():
    """Build a dictionary of ground truth labels from the pos/neg folders."""
    gt_dict = {}
    
    # Path to the positive and negative directories
    pos_dir = "/home/Desktop/choroid/final_pos_neg_test_data_merging_25_2/pos/rgb_images/"
    neg_dir = "/home/Desktop/choroid/final_pos_neg_test_data_merging_25_2/neg/rgb_images/"
    
    # Process positive tiles
    for file in os.listdir(pos_dir):
        if file.endswith('.tif'):
            tile_num = extract_tile_number(file)
            if tile_num is not None:
                gt_dict[tile_num] = 1  # 1 represents positive
    
    # Process negative tiles
    for file in os.listdir(neg_dir):
        if file.endswith('.tif'):
            tile_num = extract_tile_number(file)
            if tile_num is not None:
                gt_dict[tile_num] = 0  # 0 represents negative
    
    return gt_dict

def prepare_labels(predictions, ground_truth):
    """
    Prepare aligned lists of true labels and predictions for evaluation.
    """
    y_true = []
    y_pred = []
    
    for tile_num, pred_label in predictions.items():
        if tile_num in ground_truth:
            y_true.append(ground_truth[tile_num])
            y_pred.append(int(pred_label)) if pred_label is not None else y_pred.append(-1)
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score using sklearn."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, labels=[0, 1], average='macro')
    rec = recall_score(y_true, y_pred, zero_division=0, labels=[0, 1], average='macro')
    f1 = f1_score(y_true, y_pred, zero_division=0, labels=[0, 1], average='macro')
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plot a confusion matrix with matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Negative', 'Positive'], 
           yticklabels=['Negative', 'Positive'],
           ylabel='David labels',
           xlabel='Our GT label',
           title=title)
    
    # Add text annotations in the cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_dict, title, save_path=None):
    """
    Plot performance metrics as a bar chart.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [metrics_dict[m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)  # Set y-axis limit with some padding
    ax.set_title(title)
    ax.set_ylabel('Score')
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Create a directory for saving plots
    plots_dir = "/root/Gully_Detection/labeling_tool/v2_results/final/agg/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Build ground truth dictionary
    print("Building ground truth dictionary...")
    ground_truth = build_ground_truth_dict()
    print(f"Found {sum(1 for v in ground_truth.values() if v == 1)} positive and "
          f"{sum(1 for v in ground_truth.values() if v == 0)} negative tiles in ground truth.")
    
    # Get all majority.json files
    agg_dir = "/root/Gully_Detection/labeling_tool/v2_results/final/agg/"
    majority_files = glob.glob(os.path.join(agg_dir, "*_majority.json"))
    
    results = {}
    
    # Process each majority file
    for file_path in majority_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing {file_name}...")
        
        # Load predictions
        with open(file_path, 'r') as f:
            predictions = json.load(f)
        
        # Prepare aligned lists of true labels and predictions
        y_true, y_pred = prepare_labels(predictions, ground_truth)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Store results
        results[file_name] = {
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn
            },
            "metrics": metrics
        }
        
        # Create a proper confusion matrix for visualization
        cm_display = np.array([[tn, fp], [fn, tp]])
        
        # Plot confusion matrix
        plot_title = f"Confusion Matrix: {file_name}"
        cm_save_path = os.path.join(plots_dir, f"{os.path.splitext(file_name)[0]}_cm.png")
        plot_confusion_matrix(cm_display, plot_title, cm_save_path)
        
        # Plot metrics
        metrics_title = f"Performance Metrics: {file_name}"
        metrics_save_path = os.path.join(plots_dir, f"{os.path.splitext(file_name)[0]}_metrics.png")
        plot_metrics(metrics, metrics_title, metrics_save_path)
        
        # Print results
        print(f"Confusion Matrix for {file_name}:")
        print(f"TP: {tp}, FP: {fp}")
        print(f"FN: {fn}, TN: {tn}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}") 
        print(f"Plots saved to {plots_dir}")
    
    print(f"\nAll plots have been saved to {plots_dir}")

if __name__ == "__main__":
    main()
