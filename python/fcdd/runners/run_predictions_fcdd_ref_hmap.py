""" Runs FCDD predictions based on trained snapshot
"""
#%%
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from predictor import predict_and_evaluate_ref
from fcdd.util.logging import Logger


# Define a list of strings based on a list of integers moddifying an f-string
base_path = "../../../data/results"

#target_path = f"{base_path}/cv_maps/fcdd_task_1"
# target_path = f"{base_path}/cv_maps/fcdd_task_2"
#target_path = f"{base_path}/cv_maps/fcdd_task_0" # Used for all figures except for 2024 updates (Figures 3 and 4 with changes)

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

import pandas as pd

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

#### Figure 4
# Task #1 in Paper (task 0 in runs)

# Task # 2 in Paper (task 1 in runs)
# [644, 519, 518, 517, 1811]
# [44, 25, 48, 10, 34, 30, 22]
# Task #3 (5 year risk -- not in paper)
#  NaN
# [2, 3, 6, 21, 1, 0, 16]

#%%
BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Adjust accordingly
metadata_df = pd.read_csv(f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0)
metadata_df["Breast"] = metadata_df["StudyId"] + metadata_df["MRILaterality"]
metadata_filtered_df = metadata_df[["Age", "Breast", "BIRADS", "Final Class Benign/Malignant"]]
# Left join df with metadata_filtered_df on study_id and Breast
df = df.merge(metadata_filtered_df, left_on="study_id", right_on="Breast", how="left")

#%%
## Generating results for confusion matrix of explanation (fig 4c)
# Create new column "Prediction" by thresholding the all_scores column
task = 0
if task == 0:
    df["Prediction"] = df["all_scores"].apply(lambda x: 1 if x > 0.07 else 0) # Pick being conservative (less false negatives)

from sklearn.metrics import confusion_matrix
confusion_matrix(df["all_labels"], df["Prediction"])

#%%
# Based on the exploration of the indices above create indices lists
# tp = [0, 2, 5] # Old version
tp = [0, 2, 5, 6, 11] # New version with additional indices
# tn = [647, 654, 657] 
tn = [647, 654, 657, 670, 673] # New version with additional indices
# fp = [653, 661, 658]
fp = [653, 661, 658, 659, 660] # New version with additional indices
# fn = [92, 127, 80]
fn = [92, 127, 80, 96, 103]


# Append the indices to a list
indices = tp + tn + fp + fn

# #%%
# # In df find the rows with all_labels == 1 and Prediction == 0
# df[(df["all_labels"] == 1) & (df["Prediction"] == 0)]

#%%
def plot_images(indices, title, row, ax):
    for col, i in enumerate(indices):
        ax[row, col].imshow(plt.imread(df["all_paths"][i]), cmap="Greys_r")
        ax[row, col].set_title(f"{title} {i}")  # Using the index i itself as the title
        #ax[row, col].axis('off')  # Hide the axis

# Create a subplot layout
fig, ax = plt.subplots(4, len(tp), figsize=(15, 10))  # Adjust figure size as needed

# Plot true positives
plot_images(tp, "TP", 0, ax)
# Plot false positives
plot_images(fp, "FP", 1, ax)
# Plot true negatives
plot_images(tn, "TN", 2, ax)
# Plot false negatives
plot_images(fn, "FN", 3, ax)
plt.show()

#%%
# Create heatmap for figure 4c
trainer.heatmap_generation(
    labels = results_test["all_labels"],
    ascores = results_test["all_upsampled"],
    imgs = results_test["all_images"],
    name="fig4c_2024",
    specific_idx=([0], indices),
)


# %%
# Create figure OLD figure 4d. Go over the BIRADS scores from 0 to 6 and pick two random rows for each BIRADS score. Then plot

# birads = [0, 1, 2, 3, 4, 5, 6]
# birads_indices = []
# for b in birads:
#     # Get the indices of the rows with the BIRADS score
#     birads_index = df[df["BIRADS"] == b].index
#     # Sample two random indices from the birads_index
#     birads_indices.append(np.random.choice(birads_index, 2, replace=False))

# # Plot each of the indices with plot_images Function
# fig, ax = plt.subplots(7, 2, figsize=(15, 10))  # Adjust figure size as needed
# for row, birads_index in enumerate(birads_indices):
#     plot_images(birads_index, f"BIRADS {row}", row, ax)
# plt.show()

# #%%
# # Good indices based on task 0, ascending order [array([2795,  762]), array([3000,  856]), array([ 675, 1184]), array([2768, 2173]), array([2307,  855]), array([272, 300]), array([372, 144])]

# # Combined birads_indices into a single list
# birads_indices = [item for sublist in birads_indices for item in sublist]

# # Run predictions
# trainer.heatmap_generation(
#     labels = results_test["all_labels"],
#     ascores = results_test["all_upsampled"],
#     imgs = results_test["all_images"],
#     name="fig4d",
#     specific_idx=([0], birads_indices),
# )
# #%%

