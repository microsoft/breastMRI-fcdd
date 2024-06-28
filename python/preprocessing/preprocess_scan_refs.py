""" Takes a data folder and creates references compatible with run_scan_refs.py. """

#%%
import os
import numpy as np
import pandas as pd
from ast import literal_eval

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Path to the global data folder
metadata_df = pd.read_csv(
    f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0
)  # Metadata dataframe
RISK_PATH = (
    f"{BASE_DIR}/data/metadata/risk_cohort_granular.csv"  # 5-year risk dataframe
)
risk_df = pd.read_csv(RISK_PATH, index_col=0)


def get_all_paths(folder_path):
    all_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_paths.append(os.path.join(root, file))
    return all_paths


def create_references_sym(folder_path: str, add_zero_tiff: bool = False):
    """Creates symmetric references for a given folder, so the reference of one breast is the other breast in the same exam."""

    # Get all paths in folder
    all_paths = get_all_paths(folder_path)
    # Drop from all_paths the ones that are not .tiff
    all_paths = [path for path in all_paths if path.endswith(".tiff")]
    # Create dataframe with all_paths naming the column "Actual"
    df = pd.DataFrame(all_paths, columns=["Actual"])
    # Create column "Label" with 1 if the path contains "anomalous" and 0 otherwise
    df["Label"] = df["Actual"].apply(lambda x: 1 if "anomalous" in x else 0)
    # Create column "Split" with "train" if the path contains "train" and "test" otherwise
    df["Split"] = df["Actual"].apply(lambda x: "train" if "train" in x else "test")
    df["ID"] = df["Actual"].apply(lambda x: x.split("/")[-1].split(".")[0][:-1])
    df["Side"] = df["Actual"].apply(lambda x: x.split("/")[-1].split(".")[0][-1])
    df["OppositeSide"] = df["Side"].apply(lambda x: "R" if x == "L" else "L")
    # Combine columns "ID" and "OppositeSide" to create a column "OppositeID"
    df["OppositeID"] = df["ID"] + df["OppositeSide"]

    # Create a column "Reference" by finding in all_paths the path that contains string in OppositeID.
    # If the path is found, use the path for "Reference", otherwise set as NaN
    df["Reference"] = df["OppositeID"].apply(
        lambda x: next((path for path in all_paths if x in path), None)
    )
    # Drop all rows where Reference is NaN, these are one sided scans
    # df = df.dropna(subset=["Reference"])
    if not add_zero_tiff:
        df["Reference"] = df.apply(
            lambda x: np.random.choice(df[df["Actual"].str.contains("normal") & df["Actual"].str.contains("train")]["Actual"]) if pd.isnull(x["Reference"]) and "train" in x["Actual"] else x["Reference"],
            axis=1
        )
        df["Reference"] = df.apply(
            lambda x: np.random.choice(df[df["Actual"].str.contains("normal") & df["Actual"].str.contains("test")]["Actual"]) if pd.isnull(x["Reference"]) and "test" in x["Actual"] else x["Reference"],
            axis=1
        )
        # Drop rows that have NaN or None in Reference
        df = df.dropna(subset=["Reference"])
    if add_zero_tiff:
        zero_tiff = "/home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation_3/zero_ref.tiff"
        df["Reference"] = df.apply(
            lambda x: zero_tiff if pd.isnull(x["Reference"]) else x["Reference"],
            axis=1
        )
    # Get more readable "Reference" and "Actual" columns by just keeping the file name and the top folder name
    df["Reference_Read"] = df["Reference"].apply(
        lambda x: x.split("/")[-2] + "/" + x.split("/")[-1]
    )
    df["Actual_Read"] = df["Actual"].apply(
        lambda x: x.split("/")[-2] + "/" + x.split("/")[-1]
    )
    # Check if df["Reference_Read"] has the string "anomalous" in it, if so, set "Reference" to to random element in column Actual that has "normal" in it
    if not add_zero_tiff:
        df["Reference"] = df.apply(
            lambda x: np.random.choice(df[df["Actual"].str.contains("normal") & df["Actual"].str.contains("train")]["Actual"]) if "anomalous" in x["Reference_Read"] and "train" in x["Actual"] else x["Reference"],
            axis=1
        )
        df["Reference"] = df.apply(
            lambda x: np.random.choice(df[df["Actual"].str.contains("normal") & df["Actual"].str.contains("test")]["Actual"]) if "anomalous" in x["Reference_Read"] and "test" in x["Actual"] else x["Reference"],
            axis=1
        )
    if add_zero_tiff:
        zero_tiff = "/home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation_3/zero_ref.tiff"
        df["Reference"] = df.apply(
            lambda x: zero_tiff if "anomalous" in x["Reference_Read"] else x["Reference"],
            axis=1
        )
    # Now update Referece_Read based on Reference
    df["Reference_Read"] = df["Reference"].apply(
        lambda x: x.split("/")[-2] + "/" + x.split("/")[-1]
    )
    df2 = df.copy()
    # Remove records where "Reference_Read" contains "anomalous" as it always should be normal
    # df = df[~df["Reference_Read"].str.contains("anomalous")]
    # Note: will limit patients to those that develop cancer in only one side (TBC)
    # Create and save reference dataframes
    train_df = df[df["Split"] == "train"][["Actual", "Label", "Reference"]]
    test_df = df[df["Split"] == "test"][["Actual", "Label", "Reference"]]
    train_df.to_csv(f"{folder_path}/custom/train_ref.csv")
    test_df.to_csv(f"{folder_path}/custom/test_ref.csv")

    return train_df, test_df, df2


def create_references_past(folder_path: str, risk_df: pd.DataFrame = risk_df):
    """Creates past references for a given folder, only works for Task #2 --> The patient develops cancer after benign exam."""

    all_paths = get_all_paths(folder_path)

    # Explode individual exams into rows
    risk_df = risk_df[
        [
            "PatientId",
            "StudyCode",
            "Last_Benign",
            "Develops_Cancer",
            "Final Class Benign/Malignant",
            "MRILaterality",
        ]
    ]
    risk_df["StudyCode"] = risk_df["StudyCode"].apply(literal_eval)
    risk_df["Final Class Benign/Malignant"] = risk_df[
        "Final Class Benign/Malignant"
    ].apply(literal_eval)
    risk_df = risk_df.explode(["StudyCode", "Final Class Benign/Malignant"])
    risk_df["Previous"] = risk_df.groupby(["PatientId"])["StudyCode"].shift(1)
    risk_df["Previou_Class"] = risk_df.groupby(["PatientId"])[
        "Final Class Benign/Malignant"
    ].shift(1)
    risk_df = risk_df.dropna(subset=["Previous"])

    # Drop malignant from Previous_Class, i.e. the reference will only be normal or benign
    risk_df = risk_df[risk_df["Previou_Class"] != "Malignant"]
    # If StudyCode and Reference don't finish in the same letter, set Reference to NaN
    risk_df["Previous"] = risk_df.apply(
        lambda x: x["Previous"] if x["StudyCode"][-1] == x["Previous"][-1] else np.nan,
        axis=1,
    )
    # Drops the first scan or single scan. Note: in the future, we may want to pass a blank reference
    risk_df = risk_df.dropna(subset=["Previous"])

    # Choose all paths that end in .tiff
    all_paths = [path for path in all_paths if path.endswith(".tiff")]
    # Create df with column Actual based in all_paths
    df = pd.DataFrame(all_paths, columns=["Actual"])
    # Create column "Label" with 1 if the path contains "anomalous" and 0 otherwise
    df["Label"] = df["Actual"].apply(lambda x: 1 if "anomalous" in x else 0)
    # Create column "Split" with "train" if the path contains "train" and "test" otherwise
    df["Split"] = df["Actual"].apply(lambda x: "train" if "train" in x else "test")
    # Create a column "ID" with the file name without the extension
    df["ID"] = df["Actual"].apply(lambda x: x.split("/")[-1].split(".")[0])
    # Map the ID in df to the StudyCode in risk_df, and create a column "Reference" with the value of the previous benign scan
    df["Reference"] = df["ID"].map(risk_df.set_index("StudyCode")["Previous"])
    # Drop all rows where Reference is NaN, these do not have a clear reference
    df = df.dropna(subset=["Reference"])

    # Replace each value in Reference with the path in all_paths that contains the values in Reference
    df["Reference"] = df["Reference"].apply(
        lambda x: next((path for path in all_paths if x in path), None)
    )
    df = df.dropna(subset=["Reference"])
    # Create train and test dataframes and save them
    train_df = df[df["Split"] == "train"][["Actual", "Label", "Reference"]]
    test_df = df[df["Split"] == "test"][["Actual", "Label", "Reference"]]
    train_df.to_csv(f"{folder}/custom/train_ref.csv")
    test_df.to_csv(f"{folder}/custom/test_ref.csv")

    return train_df, test_df


# /home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation/custom
# Create references for Task #0, symm reference
# ROOT_NAME = "fccd_data_patient_kc_both_cv"  # For task #1
ROOT_NAME = "fccd_data_patient_task0_cv" # For task # 0
folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]
# ROOT_NAME = "fcdd_data_explanation" # For explanation, create reference
# folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}/"]
# # Run references creation for each folder in the CV split
for folder in folder_paths:
    train_df, test_df, df = create_references_sym(folder, add_zero_tiff=False) # Zero tiff works best
# Create references for Task #1
# # ROOT_NAME = "fccd_data_patient_kc_both_cv"  # For task #1
# ROOT_NAME = fcdd_data_patient_task0_cv
# # folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]
# # # Run references creation for each folder in the CV split
# # for folder in folder_paths:
# #     train_df, test_df = create_references_sym(folder)

#%%
# # In split test, get the PatiendID
# df3_test = df3[df3["Split"] == "test"]
# df3_test["PatientID"] = df3_test["ID"].apply(lambda x: x.split("_")[0])
# # Selecrt PatientID MRDL121
# df3_test[df3_test["PatientID"] == "MRDL121"]
# #%%
# import os

# target_path = "/home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation/custom/test/breast_img/anomalous/"
# org_path = "/home/felipeoviedoperhavec/ssdprivate/FH/data/flat_raw/"
# # org_path = "/home/felipeoviedoperhavec/ssdprivate/FH/data/flat_data/malignant/"
# final_path = "/home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation/custom/test/breast_img/normal/"

# # Read the files in the target folder. Create a new list of filenames.
# target_files = os.listdir(target_path)
# new_target_files = []

# for file in target_files:
#     # Split the filename and extension
#     filename, extension = os.path.splitext(file)

#     # Check if the last character of the filename is 'L' or 'R' and replace it accordingly
#     if filename[-1] == "L":
#         new_filename = filename[:-1] + "R" + extension
#     elif filename[-1] == "R":
#         new_filename = filename[:-1] + "L" + extension
#     else:
#         # Handle filenames that don't end with 'L' or 'R' (if needed)
#         new_filename = file

#     new_target_files.append(new_filename)

# #%%
# import shutil

# # Now, seach the filenames in new_target_files in the org_path and copy them to the final_path. If the file is not found, print a message and skip it.
# for file in new_target_files:
#     try:
#         shutil.copy(os.path.join(org_path, file), final_path)
#     except FileNotFoundError:
#         print(f"File {file} not found in {org_path}")
#         continue


#%%
# # Create past references for Task #2
# ROOT_NAME = "fccd_data_patient_rk_cv"  # For task #2
# folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]

# # Run references creation for each folder in the CV split
# for folder in folder_paths:
#     train_df, test_df = create_references_past(folder)

#%%

# # # Create purely symmetrical reference for Task #2
# ROOT_NAME = "fccd_data_patient_rk_cv_symm"  # For task #2
# folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]

# # Run references creation for each folder in the CV split
# for folder in folder_paths:
#     train_df, test_df = create_references_sym(folder)

####################

#%%
# IF needed, make copies of folders to create a new CV split
# import shutil
# ROOT_NAME = "fccd_data_patient_rk_cv"
# folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]
# for folder in folder_paths:
#     shutil.copytree(folder, folder.replace("cv", "cv_symm"))

# %%
# Create past references for Task #2 using a minimal cv split, which drops malignants from the train set as well
# ROOT_NAME = "fccd_data_patient_min_cv"  # For task #2
# folder_paths = [f"{BASE_DIR}/data/{ROOT_NAME}_{i}/" for i in range(5)]

# # Run references creation for each folder in the CV split
# for folder in folder_paths:
#     train_df, test_df = create_references_past(folder)
