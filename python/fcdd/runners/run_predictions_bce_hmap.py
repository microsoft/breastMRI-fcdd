""" Runs FCDD predictions based on trained snapshot
"""
#%%
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from predictor import predict_and_evaluate_bce
from fcdd.util.logging import Logger


# Define a list of strings based on a list of integers moddifying an f-string
base_path = "../../../data/results"

# target_path = f"{base_path}/cv_maps/bce_task_1"
# target_path = f"{base_path}/cv_maps/bce_task_2"
target_path = f"{base_path}/cv_maps/bce_task_0"
logger = Logger(target_path)
# # If target_path does not exist, create it
# if not os.path.exists(target_path):
#     os.makedirs(target_path)

# results_path = (
#     f"{base_path}/fcdd_20220922212317bce_task1_cv_0_crop_custom_/normal_0/it_2/"
# )
# results_path = (
#     f"{base_path}/fcdd_20220922210714bce_task2_cv_0_crop_custom_/normal_0/it_1/"
# )
results_path = ("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/results/fcdd_20230811001734task0_no_aug_200_bce_0_custom_/normal_0/it_0/")
results_test, trainer = predict_and_evaluate_bce(
    results_path=results_path, log_path=target_path, on_train=False
)

#%%
# Account for balance_class() operation in training set
# Create a dictionary based on results_train["all_paths"] where the keys are the positions of the paths and the values are the paths
ind_to_path = {
    i: results_test["all_paths"][i] for i in range(len(results_test["all_paths"]))
}
#%%
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
# Sort the dataframe by all_scores
# df = df.sort_values(by="all_scores")
#%%
# Create a column with study id by taking the last part of the path and removing .tiff
df["study_id"] = df["all_paths"].apply(lambda x: x.split("/")[-1].split(".")[0])

#%%


def plot_many_imgs(df, n_row=24, n_col=8, plot_type="most"):
    # Plots the least / most anomalous images
    # Plots 24*8 images --> 192 most anomalous
    _, axs = plt.subplots(n_row, n_col, figsize=(18, 48))
    df_2 = df.copy()
    df_2 = df_2.sort_values(by="all_scores")
    axs = axs.flatten()
    if plot_type == "most":
        idxs = df_2.index[-n_row * n_col :]
    elif plot_type == "least":
        idxs = df_2.index[: n_row * n_col]
    for img, ax in zip(idxs, axs):
        ax.imshow(plt.imread(df_2["all_paths"][img]), cmap="Greys_r")
        # Print the label and the fname in the title
        ax.set_title(f"{df_2['all_labels'][img]} -- {df['study_id'][img]} -- {img}")
    plt.tight_layout()
    plt.show()


# plot_many_imgs(df, plot_type="most")
plot_many_imgs(df, plot_type="least")


# Cases of interest for Task # 1, fcdd_task1_cv_0_custom_/normal_0/it_1/
# [644, 519, 518, 517, 1811]
# [44, 0, 48, 10, 51]

#%%%

trainer.heatmap_generation(
    labels=results_test["all_labels"],
    ascores=results_test["all_scores"],
    imgs=results_test["all_images"],
    name="t0",
    specific_idx=([2948, 3061, 2851, 2924, 2860, 891, 1991], [67, 49, 459, 38, 59, 17, 3298]),
    grads=results_test["all_grads"],
)

#%%
# For task # 1 (task 0 in runs)
# [67, 49, 459, 38, 59, 17, 3298]
# For task #2 (task 1 in runs)
# [44, 0, 48, 10, 51]


# # Cases of interest for Task #2
# [2, 3, 6, 21, 1, 0, 16]
#%%
BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Adjust accordingly
metadata_df = pd.read_csv(f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0)

# len("No. of images", metadata_df.PatientId.unique())
# Create column Breast merging StudyId and laterality
metadata_df["Breast"] = metadata_df["StudyId"] + metadata_df["MRILaterality"]
metadata_filtered_df = metadata_df[["Age", "Breast", "BIRADS", "Final Class Benign/Malignant"]]

# %%
# Left join df with metadata_filtered_df on study_id and Breast
df = df.merge(metadata_filtered_df, left_on="study_id", right_on="Breast", how="left")


# %%
