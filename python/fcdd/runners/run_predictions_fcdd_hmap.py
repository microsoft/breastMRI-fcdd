""" Runs FCDD predictions based on trained snapshot
"""
#%%
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from predictor import predict_and_evaluate
from fcdd.util.logging import Logger


# Define path to trained results
base_path = "../../../data/results"
# Define desired path to save heatmaps
target_path = f"{base_path}/cv_maps/figs"
logger = Logger(target_path)

task = 0

# Define path to model results and trainer
results_path = ("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/results/fcdd_20230811001734task0_no_aug_200_fcdd_0_custom_/normal_0/it_0/")
# Make predictions for the test set using the trained FCDD model
results_test, trainer = predict_and_evaluate(
    results_path=results_path, log_path=target_path, on_train=False
)

#%%
# Account for balance_class() operation in training set, which changes the order of the files
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
df = df.drop_duplicates(subset="all_paths")
df["study_id"] = df["all_paths"].apply(lambda x: x.split("/")[-1].split(".")[0])


#%%%
### Computes heatmaps for FCDD. 'specific_idx' is a tuple of lists, where the first list contains the indices of the heatmaps of interest for the normal and the second list for the abnormal class.
trainer.heatmap_generation(
    labels = results_test["all_labels"], # All labels
    ascores = results_test["all_upsampled"], # All model predicted scores
    imgs = results_test["all_images"], # All images
    name="test", # Name of the figure
    specific_idx=([2948, 3061, 2851, 2924, 2860, 891, 1991], [67, 49, 459, 38, 59, 17, 3298]), # In addition to a random sample, the model will predict specific indices in the image list
)
# %%
