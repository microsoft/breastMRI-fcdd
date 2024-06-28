""" Runs FCDD predictions based on trained snapshot
"""
#%%
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictor import predict_and_evaluate_ref
from fcdd.util.logging import Logger


# Define path to the results
base_path = "../../../data/results"

# Define task and create path for outputing results
task = 0
target_path = f"{base_path}/cv_maps/fcdd_2024_figs_ref_task_{task}"
logger = Logger(target_path)

if task == 0:
    results_path = f"/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/fcdd_20240119014001task0_ref_rand_0_ref_/normal_0/it_1/"
elif task == 1:
    results_path = f"/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/fcdd_20240122200806task1_ref_rand_0_ref_/normal_0/it_1/"

results_test, trainer = predict_and_evaluate_ref(
    results_path=results_path, log_path=target_path, on_train=False, device=4
)

# Account for balance_class() operation in training set
# Create a dictionary based on results_train["all_paths"] where the keys are the positions of the paths and the values are the paths
ind_to_path = {
    i: results_test["all_paths"][i] for i in range(len(results_test["all_paths"]))
}
# For all elements in results_train["indices"] map the index to the path using ind_to_path
results_test["exp_paths"] = [ind_to_path[i] for i in results_test["all_indices"]]

df = pd.DataFrame()
df["all_labels"] = results_test["all_labels"]
df["all_scores"] = results_test["all_scores"]
df["all_paths"] = results_test["exp_paths"]
df = df.drop_duplicates(subset="all_paths")
df["study_id"] = df["all_paths"].apply(lambda x: x.split("/")[-1].split(".")[0])

### Computes heatmaps of Figure 4a and Figure 4b
if task == 0:
    normal = [2948, 3061, 2851, 2924, 2860, 891, 1991]
    abnormal = [67, 49, 459, 38, 59, 17, 3298]
elif task == 1:
    normal = [644, 519, 518, 517, 1811]
    abnormal = [44, 25, 48, 10, 34, 30, 22]

trainer.heatmap_generation(
    labels = results_test["all_labels"],
    ascores = results_test["all_upsampled"],
    imgs = results_test["all_images"],
    name="fig4a_ref",
    specific_idx=(normal, abnormal),
)