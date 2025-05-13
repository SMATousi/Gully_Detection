#!/usr/bin/env python3
"""
Prediction Statistics Calculator

This script compares ground truth labels with model predictions and computes
various performance metrics to evaluate the model's effectiveness.

Usage:
    python prediction_statistics.py --ground_truth path/to/ground_truth.json --predictions path/to/predictions.json
"""

import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_json(file_path):
    """Load and return JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate prediction statistics by comparing ground truth and predictions")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--output_plot", required=False, help="Path to save confusion matrix plot")
    args = parser.parse_args()

    # Load data
    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)

    # Get common tile IDs
    common_ids = set(ground_truth.keys()).intersection(set(predictions.keys()))
    
    if not common_ids:
        print("Error: No common tile IDs found between ground truth and predictions")
        return
    
    # Print basic stats about the data
    print(f"Ground truth contains {len(ground_truth)} tiles")
    print(f"Predictions contain {len(predictions)} tiles")
    print(f"Common tiles: {len(common_ids)}")
    
    # Get tiles only in ground truth or predictions
    only_in_ground_truth = set(ground_truth.keys()) - set(predictions.keys())
    only_in_predictions = set(predictions.keys()) - set(ground_truth.keys())
    
    if only_in_ground_truth:
        print(f"\nTiles only in ground truth ({len(only_in_ground_truth)}): {sorted(list(only_in_ground_truth))[:10]}{'...' if len(only_in_ground_truth) > 10 else ''}")
    
    if only_in_predictions:
        print(f"\nTiles only in predictions ({len(only_in_predictions)}): {sorted(list(only_in_predictions))[:10]}{'...' if len(only_in_predictions) > 10 else ''}")
    
    # Create arrays for metrics calculation
    y_true = []
    y_pred = []
    
    # Track tile IDs by category (TP, TN, FP, FN)
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    
    # Fill arrays with data for common IDs
    for tile_id in common_ids:
        if ground_truth[tile_id] is not None:
            gt_label = int(ground_truth[tile_id])
        else:
            continue
        pred_label = int(predictions[tile_id])
        
        y_true.append(gt_label)
        y_pred.append(pred_label)
        
        # Categorize by prediction type
        if gt_label == 1 and pred_label == 1:
            true_positives.append(tile_id)
        elif gt_label == 0 and pred_label == 0:
            true_negatives.append(tile_id)
        elif gt_label == 0 and pred_label == 1:
            false_positives.append(tile_id)
        elif gt_label == 1 and pred_label == 0:
            false_negatives.append(tile_id)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\n===== Prediction Statistics =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"             Predicted Negative    Predicted Positive")
    print(f"Actual Negative    {cm[0][0]:4d} (TN)             {cm[0][1]:4d} (FP)")
    print(f"Actual Positive    {cm[1][0]:4d} (FN)             {cm[1][1]:4d} (TP)")
    
    # Display class distribution
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    print(f"\nClass Distribution in Ground Truth:")
    print(f"Positive (1): {pos_count} ({pos_count/len(y_true):.2%})")
    print(f"Negative (0): {neg_count} ({neg_count/len(y_true):.2%})")
    
    # Print example IDs for each category
    max_display = 10
    print(f"\nTrue Positives ({len(true_positives)}): {sorted(true_positives)[:max_display]}{'...' if len(true_positives) > max_display else ''}")
    print(f"True Negatives ({len(true_negatives)}): {sorted(true_negatives)[:max_display]}{'...' if len(true_negatives) > max_display else ''}")
    print(f"False Positives ({len(false_positives)}): {sorted(false_positives)[:max_display]}{'...' if len(false_positives) > max_display else ''}")
    print(f"False Negatives ({len(false_negatives)}): {sorted(false_negatives)[:max_display]}{'...' if len(false_negatives) > max_display else ''}")
    
    # Plot confusion matrix if output path is provided
    if args.output_plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative (0)', 'Positive (1)'],
                    yticklabels=['Negative (0)', 'Positive (1)'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"\nConfusion matrix plot saved to {args.output_plot}")
        
        # Also save prediction analysis as CSV
        csv_path = args.output_plot.replace('.png', '_analysis.csv').replace('.jpg', '_analysis.csv')
        analysis = []
        for tile_id in common_ids:
            if ground_truth[tile_id] is not None:
                gt_label = int(ground_truth[tile_id])
            else:
                continue
            pred_label = int(predictions[tile_id])
            category = ""
            if gt_label == 1 and pred_label == 1:
                category = "TP"
            elif gt_label == 0 and pred_label == 0:
                category = "TN"
            elif gt_label == 0 and pred_label == 1:
                category = "FP"
            elif gt_label == 1 and pred_label == 0:
                category = "FN"
            
            analysis.append({
                "Tile_ID": tile_id,
                "Ground_Truth": gt_label,
                "Prediction": pred_label,
                "Category": category
            })
        
        pd.DataFrame(analysis).to_csv(csv_path, index=False)
        print(f"Detailed prediction analysis saved to {csv_path}")

if __name__ == "__main__":
    main()
