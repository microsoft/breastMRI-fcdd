#%%
import os
import shutil
import pandas as pd

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"
BASE_FILE = "metadata"
FLAT_PATH = f"{BASE_DIR}/data/flat_raw"
REFERENCE_PATH = f"{BASE_DIR}/explanation_gt"
TARGET_PATH = f"{BASE_DIR}/data/explanation_study"
FCDD_PATH = f"{BASE_DIR}/data/fcdd_data"  # Path to FCDD

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Adjust accordingly
metadata_df = pd.read_csv(f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0)

# From REFERENCE_PATH, get a list of all the *.tiff files filenames
reference_files = [f for f in os.listdir(REFERENCE_PATH) if f.endswith(".tiff")]
# Remove the "_mask" substring from the filenames
reference_files = [f.replace("_mask", "") for f in reference_files]
# Remove .tiff extension from the filenames
reference_files = [f.replace(".tiff", "") for f in reference_files]

# %%
def create_fcdd_dir(
    train_df,
    test_df,
    grouping=None,
    dest_dir=FCDD_PATH,
    suffix=None,
):

    # Add suffix, for path consistency
    if suffix:
        dest_dir = dest_dir + f"_{suffix}"  # Suffix

    # Check if dest_dir exists, if it does, delete it, then re-create folder structure
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    # Make directory structure for FCDD
    os.makedirs(f"{dest_dir}")
    os.makedirs(f"{dest_dir}/custom/")
    os.makedirs(f"{dest_dir}/custom/train")
    os.makedirs(f"{dest_dir}/custom/test")
    os.makedirs(f"{dest_dir}/custom/train/breast_img")
    os.makedirs(f"{dest_dir}/custom/test/breast_img")
    os.makedirs(f"{dest_dir}/custom/train/breast_img/anomalous")
    os.makedirs(f"{dest_dir}/custom/test/breast_img/anomalous")
    os.makedirs(f"{dest_dir}/custom/train/breast_img/normal")
    os.makedirs(f"{dest_dir}/custom/test/breast_img/normal")

    # Copy files according to split
    copy_fcdd(train_df, dest_dir=dest_dir, df_type="train")
    copy_fcdd(test_df, dest_dir=dest_dir, df_type="test")

    # Save train and test splits
    train_df.to_csv(f"{BASE_DIR}/data/split_logs/train-{grouping}-{suffix}.csv")
    test_df.to_csv(f"{BASE_DIR}/data/split_logs/test-{grouping}-{suffix}.csv")


def copy_fcdd(df, dest_dir, df_type="train"):

    for pth, label, lat, sid in zip(df.Path, df.Label, df.Laterality, df.StudyID):
        try:
            if label == "Malignant":
                src = pth
                dest = (
                    f"{dest_dir}/custom/{df_type}/breast_img/anomalous/{sid}{lat}.tiff"
                )
                shutil.copyfile(src, dest)
            elif label == "Benign":
                src = pth
                dest = f"{dest_dir}/custom/{df_type}/breast_img/normal/{sid}{lat}.tiff"
                shutil.copyfile(src, dest)

        except:
            print(f"Error in file {pth}")
            raise


def label_dir(source_dir=FLAT_PATH, dest_dir=TARGET_PATH, metadata=metadata_df):
    """Creates directory with labelled data.
    Benign and malignant correspond to detection on image.

    Args:
        source_dir (str, optional): Source directory. Defaults to FLAT_PATH.
        dest_dir (str, optional): Destination directory. Defaults to TARGET_PATH.
        metadata (pd.DataFrame, optional): Metadata pandas dataframe. Defaults to metadata_df, the original metadata dataframe.
    Returns:
        path of target folder (str)
        dir metadata file (pd.DataFrame)
    """

    # Check if dest_dir exists, if it does, delete it, then re-create folder structure
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    # Make directories
    os.makedirs(f"{dest_dir}")
    os.makedirs(f"{dest_dir}/benign")
    os.makedirs(f"{dest_dir}/malignant")
    # Exclude flagged data
    metadata = metadata[metadata.EXCLUDED != "Y"]
    # Accumulate errors and register final metadata
    error_list = []
    dir_meta = pd.DataFrame()

    # Loop a structure dataset adequatelly
    for sid, lat, pid, label, mdt, brd, ind in zip(
        metadata.StudyId,
        metadata.MRILaterality,
        metadata.PatientId,
        metadata["Final Class Benign/Malignant"],
        metadata.MRIdate,
        metadata.BIRADS,
        metadata.Indication,
    ):
        try:
            data_dic = {}
            if label == "Malignant":
                src = f"{source_dir}/{sid}{lat}.tiff"
                dest = f"{dest_dir}/malignant/{sid}{lat}.tiff"
                shutil.copyfile(src, dest)

            elif label == "Benign":
                src = f"{source_dir}/{sid}{lat}.tiff"
                dest = f"{dest_dir}/benign/{sid}{lat}.tiff"
                shutil.copyfile(src, dest)

            data_dic["Path"] = dest
            data_dic["Label"] = label
            data_dic["Patient"] = pid
            data_dic["StudyID"] = sid
            data_dic["Laterality"] = lat
            data_dic["MRIdate"] = mdt
            data_dic["BIRADS"] = brd
            data_dic["Indication"] = ind

            # dir_meta = dir_meta.append(data_dic, ignore_index=True)
            dir_meta = pd.concat([dir_meta, pd.DataFrame(data_dic, index=[0])])

        except FileNotFoundError:
            error_list.append(src + "----" + dest)

        except RuntimeError:
            print(sid)
            print(lat)
            print(label)
            raise
    print("Total # of missing files:", len(error_list))
    dir_meta = dir_meta.dropna()

    # print(error_list)
    dir_meta.Patient = dir_meta.Patient.astype("int")

    return dest_dir, dir_meta


whole_dir, whole_df = label_dir(source_dir=FLAT_PATH, dest_dir=TARGET_PATH)
#%%
# Create SID column by merging StudyID and Laterality columns
whole_df["SID"] = whole_df.StudyID + whole_df.Laterality

# Modify whole_df
whole_df["Explanation_Group"] = whole_df.StudyID.apply(
    lambda x: 1 if x in reference_files else 0
)

# Split whole_df into train_df if Explanation_Group is 0, and test_df if Explanation Group is 1
train_df = whole_df[whole_df.Explanation_Group == 0]
test_df = whole_df[whole_df.Explanation_Group == 1]

# Get list of patients ids
test_patients = test_df.Patient.values

# If train_df.Patient is in test_patients, remove that record from train_df
train_df = train_df[~train_df.Patient.isin(test_patients)]

# Create train and test directories
create_fcdd_dir(
    train_df,
    test_df,
    grouping="Patient",
    suffix="explanation",
)


#%%
