# Copyright (c) 2021 liznerski (original FCDD work)
# Copyright (c) 2025 Microsoft Corporation (FCDD for breast cancer detection)
# Licensed under the MIT License.

""" Unified prediction script for FCDD, BCE, HSC, and FCDD_REF models
"""
import os
import json
import argparse
import pandas as pd
import torch
from predictor import predict_and_evaluate, predict_and_evaluate_bce, predict_and_evaluate_hsc, predict_and_evaluate_ref
from fcdd.util.logging import Logger


def main():
    parser = argparse.ArgumentParser(description='Run predictions with different models')
    parser.add_argument('--model', type=str, choices=['fcdd', 'bce', 'hsc', 'fcdd_ref'], 
                       default='fcdd', help='Model type to use for prediction')
    parser.add_argument('--task', type=int, choices=[1, 2], default=1, 
                       help='Task number (1 or 2)')
    parser.add_argument('--snapshot_path', type=str, default=None,
                       help='Path to model snapshot directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device number')
    
    args = parser.parse_args()
    
    # Define default paths based on model and task
    default_paths = {
        'fcdd': {
            1: "../../data/fcdd_20230811001734task0_no_aug_200_fcdd_0_custom_/normal_0/it_0/",
            2: "../../data/fcdd_20230811205123task1_no_aug_200_fcdd_0_custom_/normal_0/it_0/"
        },
        'bce': {
            1: "../../data/fcdd_20230811001734task0_no_aug_200_bce_0_custom_/normal_0/it_0/",
            2: "../../data/fcdd_20230812180854task1_no_aug_200_bce_0_custom_/normal_0/it_0/"
        },
        'hsc': {
            1: "../../data/fcdd_20230813151957task0_no_aug_200_hsc_0_custom_/normal_0/it_0/",
            2: "../../data/fcdd_20230813151957task1_no_aug_200_hsc_0_custom_/normal_0/it_0/"
        },
        'fcdd_ref': {
            1: "../../data/fcdd_20240119014001task0_ref_rand_0_ref_/normal_0/it_1/",
            2: "../../data/fcdd_20240122200806task1_ref_rand_0_ref_/normal_0/it_1/"
        }
    }
    
    # Set snapshot path
    snapshot_path = args.snapshot_path or default_paths[args.model][args.task]
    
    # Set output directory
    base_path = "../../data"
    if args.output_dir:
        target_path = args.output_dir
    else:
        target_path = f"{base_path}/results/{args.model}_task_{args.task}"
    
    os.makedirs(target_path, exist_ok=True)
    logger = Logger(target_path)
    
    print(f"Running {args.model.upper()} predictions")
    print(f"Snapshot path: {snapshot_path}")
    print(f"Output path: {target_path}")
    
    # Run predictions based on model type
    if args.model == 'fcdd':
        results_test, trainer = predict_and_evaluate(
            results_path=snapshot_path, 
            log_path=target_path, 
            on_train=False,
            device=args.device
        )
    elif args.model == 'bce':
        results_test, trainer = predict_and_evaluate_bce(
            results_path=snapshot_path, 
            log_path=target_path, 
            on_train=False,
            device=args.device
        )
    elif args.model == 'hsc':
        results_test, trainer = predict_and_evaluate_hsc(
            results_path=snapshot_path, 
            log_path=target_path, 
            on_train=False,
            device=args.device
        )
    elif args.model == 'fcdd_ref':
        results_test, trainer = predict_and_evaluate_ref(
            results_path=snapshot_path, 
            log_path=target_path, 
            on_train=False,
            device=args.device
        )
    
    print("Predictions completed.")
    
    # Process results - account for balance_class() operation
    ind_to_path = {
        i: results_test["all_paths"][i] for i in range(len(results_test["all_paths"]))
    }
    results_test["exp_paths"] = [ind_to_path[i] for i in results_test["all_indices"]]
    
    # Create DataFrame for analysis
    df = pd.DataFrame()
    df["all_labels"] = results_test["all_labels"]
    df["all_scores"] = results_test["all_scores"]
    df["all_paths"] = results_test["exp_paths"]
    df = df.drop_duplicates(subset="all_paths")
    df["study_id"] = df["all_paths"].apply(lambda x: x.split("/")[-1].split(".")[0])
    
    print(f"ROC AUC: {results_test.get('roc_auc', 'N/A')}")
    print(f"PR AUC: {results_test.get('pr_auc', 'N/A')}")
    
    # Always save predictions
    results_to_save = {
        "model": args.model,
        "task": args.task,
        "roc_auc": results_test.get("roc_auc"),
        "pr_auc": results_test.get("pr_auc"),
        "all_scores": results_test["all_scores"],
        "all_labels": results_test["all_labels"],
        "all_paths": results_test["exp_paths"]
    }
    
    with open(f"{target_path}/predictions_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"Predictions saved to: {target_path}/predictions_results.json")
    
    # Generate heatmaps
    print("Computing heatmaps...")
    
    # Prepare data for heatmap generation based on model type
    if args.model == 'fcdd':
        trainer.heatmap_generation(
            labels=results_test["all_labels"],
            ascores=results_test["all_upsampled"],
            imgs=results_test["all_images"],
            name=f"{args.model}_task_{args.task}_heatmaps",
        )
    elif args.model in ['bce', 'hsc']:
        # Convert scores to tensor and add dimension for compatibility
        all_scores = torch.tensor(results_test["all_scores"]).unsqueeze(-1)
        trainer.heatmap_generation(
            labels=results_test["all_labels"],
            ascores=all_scores,
            imgs=results_test["all_images"],
            name=f"{args.model}_task_{args.task}_heatmaps",
            grads=results_test["all_grads"],
        )
    elif args.model == 'fcdd_ref':
        trainer.heatmap_generation(
            labels=results_test["all_labels"],
            ascores=results_test["all_upsampled"],
            imgs=results_test["all_images"],
            name=f"{args.model}_task_{args.task}_heatmaps",
        )
    
    print("Heatmap generation completed.")
    print(f"Results saved to: {target_path}")


if __name__ == "__main__":
    main()
