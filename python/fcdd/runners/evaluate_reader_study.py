""" Inference and heatmap generation example for BCE, HSC and FCDD models
"""
#%%
import torch
import pandas as pd
from torch.utils.data import DataLoader
from fcdd.util.logging import Logger

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from predictor import (
    predict_and_evaluate,
    predict_and_evaluate_bce,
    predict_and_evaluate_hsc,
    predict_and_evaluate_ref
)
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# Select gpu 0 as the default device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define location of metadata (for filenames or clinical variables)
BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Adjust accordingly
metadata_df = pd.read_csv(f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0)

# Define path of ground truth annotations [Optional]
gt_path = "/home/felipeoviedoperhavec/ssdprivate/FH/explanation_gt"

# Load images from gt_path using DataLoader and ImageFolder
gt_dataset = ImageFolder(gt_path, transform=transform)
gt_loader = DataLoader(gt_dataset, batch_size=len(gt_dataset), shuffle=False)
gt_masks, labels = next(iter(gt_loader))
gt_filenames = [gt_dataset.samples[i][0] for i in range(len(gt_dataset))]

# %%
# Define paths of results (contains trained models)
base_path = "/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/"

#%%
# Make BCE predictions
target_path = f"{base_path}/cv_maps/explanation_bce"
logger_bce = Logger(target_path)
results_path = f"{base_path}/fcdd_20230315204230bce_explanation_custom_/normal_0/it_0/"

results_bce, trainer_bce = predict_and_evaluate_bce(
    results_path=results_path, log_path=target_path, on_train=False
)
#%%
# Make FCDD_REF predictions
target_path = f"{base_path}/cv_maps/explanation_ref"
logger_ref = Logger(target_path)
results_path = f"{base_path}/fcdd_20240323040022fcdd_explanation_ref_ref_/normal_0/it_0/"

results_fcdd_ref, trainer_fcdd_ref = predict_and_evaluate_ref(
    results_path=results_path, log_path=target_path, on_train=False
)
# %%
# Make FCDD predictions
target_path = f"{base_path}/cv_maps/explanation_fcdd"
logger_fcdd = Logger(target_path)
results_path = f"{base_path}/fcdd_20230314222211fcdd_explanation_custom_/normal_0/it_0/"

results_fcdd, trainer_fcdd = predict_and_evaluate(
    results_path=results_path, log_path=target_path, on_train=False
)
#%%
# Make HSC predictions
target_path = f"{base_path}/cv_maps/explanation_hsc"
logger_hsc = Logger(target_path)

results_path = f"{base_path}/fcdd_20230316040428hsc_explanation_custom_/normal_0/it_0/"

results_hsc, trainer_hsc = predict_and_evaluate_hsc(
    results_path=results_path, log_path=target_path, on_train=False
)
#%%
# Get filenames from results_bce["all_paths"][:-2] by taking the last element in path
results_filenames = [path.split("/")[-1] for path in results_bce["all_paths"][:-2]]
# Repeat for results_fcdd
results_filenames_fcdd = [
    path.split("/")[-1] for path in results_fcdd["all_paths"][:-2]
]
# Repeat same for gt_filenames
gt_filenames = [path.split("/")[-1] for path in gt_filenames]
# Remove the "_mask" substring from gt_filenames
gt_filenames = [path.replace("_mask", "") for path in gt_filenames]
# Check if results_filenames is equal to gt_filenames
#assert results_filenames == gt_filenames
#assert results_filenames_fcdd == gt_filenames
# Find difference between results_filenames and gt_filenames
diff = list(set(results_filenames) - set(gt_filenames))

# #assert results_bce["all_paths"][:-2] == results_fcdd["all_paths"][:-2]
# assert results_bce["all_paths"][:-2] == gt_filenames

# %%

# Compute ROCAUC and PR-AUC for an each pixel of image

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor


def compute_scores(img: Tensor, labels: Tensor, title: str = None, plot: bool = False):
    # Turn image into 1D list
    img = img.flatten()
    labels = labels.flatten()
    # Compute ROC-AUC
    roc_auc = roc_auc_score(labels, img)
    # Compute PR-AUC
    pr_auc = average_precision_score(labels, img)
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, img)
    # Compute PR curve
    precision, recall, _ = precision_recall_curve(labels, img)
    if plot:
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.show()

    return roc_auc, pr_auc, fpr, tpr, precision, recall


#%%
# For loop to compute scores for each image

image_scores_bce = {}

for gt, pred, fname in zip(gt_masks, results_bce["all_grads"][:-2], gt_filenames):

    roc_auc, pr_auc, fpr, tpr, precision, recall = compute_scores(pred, gt, fname)
    image_scores_bce[fname] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }

results_df = pd.DataFrame()
results_df["Study_ID"] = gt_filenames
# Get array of all roc_auc and pr_auc scores
results_df["AUC_BCE"] = [
    image_scores_bce[fname]["roc_auc"] for fname in image_scores_bce
]
results_df["PR_BCE"] = [image_scores_bce[fname]["pr_auc"] for fname in image_scores_bce]

#%%
# For loop to compute scores for each image

image_scores_hsc = {}

for gt, pred, fname in zip(gt_masks, results_hsc["all_grads"][:-2], gt_filenames):

    roc_auc, pr_auc, fpr, tpr, precision, recall = compute_scores(pred, gt, fname)
    image_scores_hsc[fname] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }
# Get array of all roc_auc and pr_auc scores
results_df["AUC_HSC"] = [
    image_scores_hsc[fname]["roc_auc"] for fname in image_scores_hsc
]
results_df["PR_HSC"] = [image_scores_hsc[fname]["pr_auc"] for fname in image_scores_hsc]

# %%

# Repeat prediction loop for image_scores_fcdd

image_scores_fcdd = {}

for gt, pred, fname in zip(gt_masks, results_fcdd["all_upsampled"][:-2], gt_filenames):

    roc_auc, pr_auc, fpr, tpr, precision, recall = compute_scores(pred, gt, fname)
    image_scores_fcdd[fname] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }

results_df["AUC_FCDD"] = [
    image_scores_fcdd[fname]["roc_auc"] for fname in image_scores_fcdd
]
results_df["PR_FCDD"] = [
    image_scores_fcdd[fname]["pr_auc"] for fname in image_scores_fcdd
]

#%%
# Match indices of fcdd_ref to all other indices
# Get fnames
fnames_ref = results_fcdd_ref["all_fnames"]
# Append .tiff to fnames_ref
fnames_ref = [f"{fname}.tiff" for fname in fnames_ref]
# Now since each fnames_ref is matched to an element in all_upsampled, reorder elements in all_upsampled so fnames_ref matches gt_filenames
all_upsampled_ref = [results_fcdd_ref["all_upsampled"][fnames_ref.index(fname)] for fname in gt_filenames]
# Concatenate list to a tensor
all_upsampled_ref = torch.cat(all_upsampled_ref).unsqueeze(1)
# # Repeat the same for results_fcdd_ref["all_images"]
all_images_ref = [results_fcdd_ref["all_images"][fnames_ref.index(fname)] for fname in gt_filenames if fname in fnames_ref]
# # Transform all_images_ref to a tensor
all_images_ref = torch.stack(all_images_ref)
# # Repeat the same for results_fcdd_ref["all_labels"]
all_labels_ref = [results_fcdd_ref["all_labels"][fnames_ref.index(fname)] for fname in gt_filenames if fname in fnames_ref]
#%%

image_scores_fcdd_ref = {}

for gt, pred, fname in zip(gt_masks, all_upsampled_ref, gt_filenames):

    roc_auc, pr_auc, fpr, tpr, precision, recall = compute_scores(pred, gt, fname)
    image_scores_fcdd_ref[fname] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }
results_df["AUC_FCDD_REF"] = [
    image_scores_fcdd_ref[fname]["roc_auc"] for fname in image_scores_fcdd_ref
]
results_df["PR_FCDD_REF"] = [
    image_scores_fcdd_ref[fname]["pr_auc"] for fname in image_scores_fcdd_ref
]
#%%
# Sort results_df by AUC_FCDD column
results_df.sort_values(by=["AUC_FCDD"], ascending=False, inplace=True)

#%%
# Using seaborn, compare with a distributions of AUC_BCE and AUC_FCDD columns in the same plot. Set max value for x axis as 1
import seaborn as sns

sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
palette = sns.color_palette("Set2")
color_dict = {"AUC_HSC": palette[2], "AUC_FCDD": palette[1], "AUC_BCE": palette[0]}

ax = sns.boxplot(
    data=results_df[["AUC_BCE", "AUC_HSC", "AUC_FCDD"]],
    orient="v",
    # palette="Set2",
    showfliers=False,
    palette=color_dict,
)
ax = sns.swarmplot(
    data=results_df[["AUC_BCE", "AUC_HSC", "AUC_FCDD"]], color="black", alpha=0.3
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# Change xticklabels to "BCE (Baseline)" "HSC" and "FCDD"
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

plt.savefig("../post/figure_5a_v2.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5a_v2.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
# Using seaborn, compare with a distributions of AUC_BCE and AUC_FCDD columns in the same plot. Set max value for x axis as 1
import seaborn as sns

sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
palette = sns.color_palette("Set2")
color_dict = {"AUC_HSC": palette[2], "AUC_FCDD": palette[1], "AUC_BCE": palette[0], "AUC_FCDD_REF": palette[3]}

ax = sns.boxplot(
    data=results_df[["AUC_BCE", "AUC_HSC", "AUC_FCDD", "AUC_FCDD_REF"]],
    orient="v",
    showfliers=False,
    palette=color_dict,
)
ax = sns.swarmplot(
    data=results_df[["AUC_BCE", "AUC_HSC", "AUC_FCDD", "AUC_FCDD_REF"]], color="black", alpha=0.3
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# Change xticklabels to "BCE (Baseline)" "HSC" and "FCDD"
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD", "FCDD\nSYMMETRIC"])

plt.savefig("../post/figure_5a_v2_SI_REF.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5a_v2_SI_REF.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
from scipy.stats import wilcoxon

bce_fcdd = wilcoxon(results_df["AUC_BCE"], results_df["AUC_FCDD"])
print("Wilcoxon BCE vs FCDD")
print(bce_fcdd)
bce_hsc = wilcoxon(results_df["AUC_BCE"], results_df["AUC_HSC"])
print("Wilcoxon BCE vs HSC")
print(bce_hsc)
print("Wilcoxon HSC vs FCDD")
hsc_fcdd = wilcoxon(results_df["AUC_HSC"], results_df["AUC_FCDD"])
print(hsc_fcdd)
bce_fcdd_ref = wilcoxon(results_df["AUC_BCE"], results_df["AUC_FCDD_REF"])
print("Wilcoxon BCE vs FCDD_REF")
print(bce_fcdd_ref)
# %%
# Load additional metadata for stratified figure
metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55.csv")
metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55_v2.csv") # Newer version with stage Ti in addition to T3
metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55_v3.csv") # Newer version with correcte histology and mass/NME and comments column

metadata_df["Full_Id"] = metadata_df["StudyId"] + metadata_df["MRILaterality"]
# %%
metadata_ref = metadata_df[["Full_Id", "BIRADS"]]
#%%
# metadata_ref["BIRADS"] = metadata_ref["BIRADS"].astype(int)
# In results_df create new column called Full_Id by dropping .tiff extension from Study_ID column
results_df["Full_Id"] = results_df["Study_ID"].str.replace(".tiff", "")
# Left join results_df and metadata_df based on Full_Id column
results_df = results_df.merge(metadata_ref, on="Full_Id", how="left")
# Transform BIRADS to interger
results_df["BIRADS"] = results_df["BIRADS"].astype(int)
#%%
# Rename Patient.and.Laterality column to Full_Id
metadata_n55.rename(columns={"Patient.and.Laterality": "Full_Id"}, inplace=True)
# Left join results_df and metadata_n55 based on Full_Id column
results_df = results_df.merge(metadata_n55, on="Full_Id", how="left")
# Create new column "Histology" that is "DCIS" if Histology_DCIS is 1 and "Invasive" if Histology_Invasive is 1
results_df["Histology"] = results_df.apply(
    lambda x: "DCIS" if x["Histology_DCIS"] == 1 else "Invasive" if x["Histology_Invasive"]==1 else None, axis=1
)
# Create new column "Histology_2" that is "DCIS" if Histology_DCIS is 1 and "Invasive" if Histology_IDC is 1
results_df["Histology_2"] = results_df.apply(
    lambda x: "DCIS" if x["Histology_DCIS"] == 1 else "IDC" if x["Histology_IDC"]==1 else None, axis=1
)
# Create new column "LesionType" that is "Mass" if LesionType_Mass is 1 and "NME" if LesionType_NME is 1
results_df["LesionType"] = results_df.apply(
    lambda x: "Mass" if x["LesionType_Mass"] == 1 else "NME" if x["LesionType_NME"]==1 else None, axis=1
)
# %%
# Melt results_df so that it can be plotted as boxplot with hue BIRADS
results_df_melted = results_df.melt(
    id_vars=["Tstage"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Tstage")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    palette="Blues",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:2] + handles[2:4][::-1], labels=labels[0:2] + labels[2:4][::-1], title="Stage") # Use for metadata_n55_v2
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])
# Set legend outside
plt.savefig("../post/figure_5b_v3_SI_w_Ti.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5b_v3_SI_w_Ti.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
# Copy results_df
results_df_filtered = results_df.copy()
# Remove rows that have Tstage as Tis
results_df_filtered = results_df_filtered[results_df_filtered["Tstage"] != "Tis"]

# Melt results_df so that it can be plotted as boxplot with hue BIRADS
results_df_melted = results_df_filtered.melt(
    id_vars=["Tstage"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
results_df_melted["Tstage"].value_counts()
#%%
results_df_melted["Tstage"] = results_df_melted["Tstage"].map(
    {
        "T1": "T1 (n=26)",
        "T2": "T2 (n=10)",
        "T3": "T3 (n=5)",
    }
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Tstage")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    color="black",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:-3], labels=labels[0:-3], title="Stage")
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

plt.savefig("../post/figure_5b_v2_Ti_removed.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5b_v2_Ti_removed.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
results_df_melted = results_df.melt(
    id_vars=["Tstage"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Tstage")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="Tstage",
    data=results_df_melted,
    # palette="Blues",
    color="black",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:-3], labels=labels[0:-3], title="Stage")
ax.legend(handles=handles[0:-4], labels=labels[0:-4], title="Stage")

ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

# plt.savefig("../post/figure_5b_v2.png", dpi=300, bbox_inches="tight")
# plt.savefig("../post/figure_5b_v2.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
results_df_melted = results_df.melt(
    id_vars=["Histology"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
results_df_melted["Histology"].value_counts()
#%%
results_df_melted["Histology"] = results_df_melted["Histology"].map(
    {
        "DCIS": "DCIS (n=14)",
        "Invasive": "Invasive (n=39)",
    }
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="Histology",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Histology")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="Histology",
    data=results_df_melted,
    # palette="Blues",
    color="black",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:-2], labels=labels[0:-2], title="Histology")
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

plt.savefig("../post/figure_5b_v2_SI_Hist.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5b_v2_SI_Hist.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
results_df_melted = results_df.melt(
    id_vars=["Histology_2"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
# Replace the values in histology_2 with value_counts. For example. DCIS : DCIS (n=10) for this you can use the results of value_counts as a map
results_df_melted["Histology_2"].value_counts()
#%%
results_df_melted["Histology_2"] = results_df_melted["Histology_2"].map(
    {
        "DCIS": "DCIS (n=14)",
        "IDC": "IDC (n=31)",
    }
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="Histology_2",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Histology")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="Histology_2",
    data=results_df_melted,
    # palette="Blues",
    color="black",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.4, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:-2], labels=labels[0:-2], title="Histology")
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

plt.savefig("../post/figure_5b_v2_SI_Hist_2.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5b_v2_SI_Hist_2.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
results_df_melted = results_df.melt(
    id_vars=["LesionType"],
    value_vars=["AUC_BCE", "AUC_HSC", "AUC_FCDD"],
    var_name="Model",
    value_name="AUC",
)
results_df_melted["LesionType"].value_counts()
# Find row with lowest AUC in results_df_melted
# results_df_melted.loc[results_df_melted["AUC"].idxmin()]
#%%
# Replace the values in lesion type with value_counts. For example. Mass : Mass (n=10) for this you can use the results of value_counts as a map
results_df_melted["LesionType"] = results_df_melted["LesionType"].map(
    {
        "Mass": "Mass (n=35)",
        "NME": "NME (n=19)",
    }
)
# Create boxplot with seaborn with legend outside of plot
sns.set(
    font_scale=1.2,
    context="poster",
    style="white",
    rc={"figure.figsize": (8, 8)},
    font="Serif",
)
ax = sns.boxplot(
    x="Model",
    y="AUC",
    hue="LesionType",
    data=results_df_melted,
    palette="Set2_r",
    showfliers=False,
    dodge=True,
    # legend=False,
)
sns.move_legend(ax, "lower right", title="Lesion Type")
# Add legend horizontally to top of plot, outside of plot
sns.swarmplot(
    x="Model",
    y="AUC",
    hue="LesionType",
    data=results_df_melted,
    color="black",
    alpha=0.5,
    dodge=True,
)
ax.set_ylim(0.2, 1.05)
# ax.set_title("Mean pixel-wise ROC-AUC for each explanation case")
ax.set_ylabel("Pixel-wise AUROC")
ax.set_xlabel("Model")
# ax.set_xticklabels(["Baseline - BCE", "Deep Anomaly Detection"])
# Hide from legend the last 3 labels and show legend title as "BIRADS"
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:-2], labels=labels[0:-2], title="Lesion Type", loc="lower right")
ax.set_xticklabels(["BCE\n(Baseline)", "HSC", "FCDD"])

plt.savefig("../post/figure_5b_v2_SI_LesionType.png", dpi=300, bbox_inches="tight")
plt.savefig("../post/figure_5b_v2_SI_LesionType.svg", dpi=300, bbox_inches="tight")
plt.show()

#%%
res_path = "/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/reader_study"
results_df.to_csv(f"{res_path}/results_reader.csv", index=False)
#%%
# Assuming img_tensor is a PyTorch tensor with dimensions (num_images, 3, 244, 244)
num_images = gt_masks.shape[0]
zero_images = torch.zeros((1, 224, 224)).unsqueeze(0).repeat(num_images + 29, 1, 1, 1) # Add 29 for 29 normal images in addition to abnormal.
gt_masks_ext = zero_images.clone()
gt_masks_ext[:num_images, :, :, :] = gt_masks

#%%
# Old Figure 5b
#idxs = [2, 4, 25, 50, 31, 23]
idxs = [41, 33, 51, 29, 13, 23, 2] # In original draft
# idxs = [45, 13, 28, 4, 31, 4, 29, 48] # In Rahul's book

# Transfor results_bce["all_scores"] list to a tensor
all_scores = torch.tensor(results_bce["all_scores"]).unsqueeze(-1)
# 
# Account for balance_class() operation in training set
# Create a dictionary based on results_train["all_paths"] where the keys are the positions of the paths and the values are the paths
ind_to_path = {
    i: results_bce["all_paths"][i] for i in range(len(results_bce["all_paths"]))
}
# For all elements in results_train["indices"] map the index to the path using ind_to_path
results_bce["exp_paths"] = [ind_to_path[i] for i in results_bce["all_indices"]]

trainer_bce.heatmap_generation(
    labels=results_bce["all_labels"],
    ascores=all_scores,
    imgs=results_bce["all_images"],
    name="tt_test",
    specific_idx=([0], idxs),
    grads=results_bce["all_grads"],
    gtmaps=gt_masks_ext,
)
# Account for balance_class() operation in training set
# Create a dictionary based on results_train["all_paths"] where the keys are the positions of the paths and the values are the paths
ind_to_path = {
    i: results_fcdd["all_paths"][i] for i in range(len(results_fcdd["all_paths"]))
}
# For all elements in results_train["indices"] map the index to the path using ind_to_path
results_fcdd["exp_paths"] = [ind_to_path[i] for i in results_fcdd["all_indices"]]

trainer_fcdd.heatmap_generation(
    labels=results_fcdd["all_labels"],
    ascores=results_fcdd["all_upsampled"],
    imgs=results_fcdd["all_images"],
    name="tt_test",
    specific_idx=([0], idxs),
    gtmaps=gt_masks_ext,
)
#%%
# New figure 5b
# Load additional metadata for stratified figure
#metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55.csv")
metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55_v2.csv")
metadata_n55 = pd.read_csv("/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/data/metadata_n55_v3.csv") # Newer version with correcte histology and mass/NME and comments column

# Rename Patient.and.Laterality column to Full_Id
metadata_n55.rename(columns={"Patient.and.Laterality": "Full_Id"}, inplace=True)
# Select columns Full_Id and Tstage from metadata_n55
metadata_n55 = metadata_n55[["Full_Id", "Tstage"]]
# Create new dataframe based on list results_fcdd["exp_paths"]
exp_paths = pd.DataFrame()
exp_paths["Paths"] = results_fcdd["all_paths"]
# Create Study_ID column by splitting the Paths column and taking the last element with .tiff extension
exp_paths["Full_Id"] = exp_paths["Paths"].apply(lambda x: x.split("/")[-1].split(".")[0])
# Left merge exp_paths and metadata_n55 based on Full_Id column
exp_paths = exp_paths.merge(metadata_n55, on="Full_Id", how="left")

#%%
# Print Tstage for each idx in idxs
for idx in idxs:
    print(exp_paths.loc[idx, "Tstage"])

# Find exp_paths with Tstage 3
exp_paths[exp_paths["Tstage"] == "T2"]

# %%
# Figure 5b_New_2024
# idxs = [41, 33, 51, 29, 13, 23, 2] # In original draft
idxs_t1 = [33, 13, 23, 51, 4, 44, 49] # Before Ti's
idxs_t1 = [33, 13, 22, 51, 4, 44, 49] # Removing Ti's, replacing with similar case
idxs_t2 = [11, 25, 29, 15]
idxs_t2 = [42, 25, 29, 15] # Removing Ti's, replacing with similar case
idxs_t3 = [2, 7, 40, 47]
# Transfor results_bce["all_scores"] list to a tensor
all_scores = torch.tensor(results_bce["all_scores"]).unsqueeze(-1)

# # For T1
trainer_bce.heatmap_generation(
    labels=results_bce["all_labels"],
    ascores=all_scores,
    imgs=results_bce["all_images"],
    name="fig_5b_2024_t1_Ti_removed",
    specific_idx=([0], idxs_t1),
    grads=results_bce["all_grads"],
    gtmaps=gt_masks_ext,
)

trainer_fcdd.heatmap_generation(
    labels=results_fcdd["all_labels"],
    ascores=results_fcdd["all_upsampled"],
    imgs=results_fcdd["all_images"],
    name="fig_5b_2024_t1_Ti_removed",
    specific_idx=([0], idxs_t1),
    gtmaps=gt_masks_ext,
)



trainer_fcdd_ref.heatmap_generation(
    labels=all_labels_ref,
    ascores=all_upsampled_ref,
    imgs=all_images_ref,
    name="fig_5b_2024_t1_SI",
    specific_idx=([0], idxs_t1),
    gtmaps=gt_masks_ext,
)

# For T2
trainer_bce.heatmap_generation(
    labels=results_bce["all_labels"],
    ascores=all_scores,
    imgs=results_bce["all_images"],
    name="fig_5b_2024_t2_Ti_removed",
    specific_idx=([0], idxs_t2),
    grads=results_bce["all_grads"],
    gtmaps=gt_masks_ext,
)

trainer_fcdd.heatmap_generation(
    labels=results_fcdd["all_labels"],
    ascores=results_fcdd["all_upsampled"],
    imgs=results_fcdd["all_images"],
    name="fig_5b_2024_t2_Ti_removed",
    specific_idx=([0], idxs_t2),
    gtmaps=gt_masks_ext,
)

trainer_fcdd_ref.heatmap_generation(
    labels=all_labels_ref,
    ascores=all_upsampled_ref,
    imgs=all_images_ref,
    name="fig_5b_2024_t2_SI",
    specific_idx=([0], idxs_t2),
    gtmaps=gt_masks_ext,
)

# For T3
trainer_bce.heatmap_generation(
    labels=results_bce["all_labels"],
    ascores=all_scores,
    imgs=results_bce["all_images"],
    name="fig_5b_2024_t3",
    specific_idx=([0], idxs_t3),
    grads=results_bce["all_grads"],
    gtmaps=gt_masks_ext,
)

trainer_fcdd.heatmap_generation(
    labels=results_fcdd["all_labels"],
    ascores=results_fcdd["all_upsampled"],
    imgs=results_fcdd["all_images"],
    name="fig_5b_2024_t3",
    specific_idx=([0], idxs_t3),
    gtmaps=gt_masks_ext,
)

trainer_fcdd_ref.heatmap_generation(
    labels=all_labels_ref,
    ascores=all_upsampled_ref,
    imgs=all_images_ref,
    name="fig_5b_2024_t3_SI",
    specific_idx=([0], idxs_t3),
    gtmaps=gt_masks_ext,
)

# %%
all_scores = torch.tensor(results_hsc["all_scores"]).unsqueeze(-1)

# For t1
trainer_hsc.heatmap_generation(
    labels=results_hsc["all_labels"],
    ascores=all_scores,
    imgs=results_hsc["all_images"],
    name="fig_5b_2024_t1_Ti_removed",
    specific_idx=([0], idxs_t1),
    gtmaps=gt_masks_ext,
    grads=results_hsc["all_grads"],
)

# For t2
trainer_hsc.heatmap_generation(
    labels=results_hsc["all_labels"],
    ascores=all_scores,
    imgs=results_hsc["all_images"],
    name="fig_5b_2024_t2_Ti_removed",
    specific_idx=([0], idxs_t2),
    gtmaps=gt_masks_ext,
    grads=results_hsc["all_grads"],
)

# For t3
trainer_hsc.heatmap_generation(
    labels=results_hsc["all_labels"],
    ascores=all_scores,
    imgs=results_hsc["all_images"],
    name="fig_5b_2024_t3_Ti_removed",
    specific_idx=([0], idxs_t3),
    gtmaps=gt_masks_ext,
    grads=results_hsc["all_grads"],
)

# %%
