""" Runs FCDD predictions based on trained snapshot
"""
#%%
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from predictor import predict_and_evaluate_hsc
from fcdd.util.logging import Logger


# Path to the results
base_path = "../../../data/results"

# Create path for outputing results
target_path = f"{base_path}/cv_maps/hsc_task_1"
logger = Logger(target_path)

# Path to specific training results and model
results_path = ("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/results/fcdd_20230813151957task0_no_aug_200_hsc_0_custom_/normal_0/it_0/")
results_test, trainer = predict_and_evaluate_hsc(
    results_path=results_path, log_path=target_path, on_train=False
)

# Account for balance_class() operation in training set
# Create a dictionary based on results_train["all_paths"] where the keys are the positions of the paths and the values are the paths
ind_to_path = {
    i: results_test["all_paths"][i] for i in range(len(results_test["all_paths"]))
}
# For all elements in results_train["indices"] map the index to the path using ind_to_path
results_test["exp_paths"] = [ind_to_path[i] for i in results_test["all_indices"]]

#%%
import pandas as pd

df = pd.DataFrame()
df["all_labels"] = results_test["all_labels"]
df["all_scores"] = results_test["all_scores"]
df["all_paths"] = results_test["exp_paths"]

#%%
# Drop all rows with repeated paths
df = df.drop_duplicates(subset="all_paths")

# Create a column with study id by taking the last part of the path and removing .tiff
df["study_id"] = df["all_paths"].apply(lambda x: x.split("/")[-1].split(".")[0])

#%%
#%%%
import torch
all_scores = torch.tensor(results_test["all_scores"]).unsqueeze(-1)

# For task 0
trainer.heatmap_generation(
    labels=results_test["all_labels"],
    ascores=all_scores,
    imgs=results_test["all_images"],
    name="test_t0",
    specific_idx=([2948, 3061, 2851, 2924, 2860, 891, 1991], [67, 49, 459, 38, 59, 17, 3298]),
    grads=results_test["all_grads"],
)

# For task 1
# trainer.heatmap_generation(
#     labels=results_test["all_labels"],
#     ascores=all_scores,
#     imgs=results_test["all_images"],
#     name="t1",
#     specific_idx=([644, 519, 518, 517, 1811], [44, 25, 48, 10, 34, 30, 22]),
#     grads=results_test["all_grads"],
# )

# %%
