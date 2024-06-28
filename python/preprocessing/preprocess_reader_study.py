#%%
import os
import shutil
from sys import meta_path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"
BASE_FILE = "metadata"
FLAT_PATH = f"{BASE_DIR}/data/flat_raw"
TARGET_PATH = f"{BASE_DIR}/data/reader_study_expanded_2"


meta_df = pd.read_csv(f"{BASE_DIR}/data/metadata/{BASE_FILE}.csv", index_col=0)
meta_df = meta_df[meta_df.BIRADS > 3.0]
meta_df = meta_df[meta_df["Final Class Benign/Malignant"] == "Malignant"]
meta_df = meta_df[meta_df.Indication == "Screening"]
meta_df = meta_df.drop_duplicates(subset=["PatientId"])


def label_dir(source_dir=FLAT_PATH, dest_dir=TARGET_PATH, metadata=meta_df):
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
    # Subsample metadata
    meta_4f = metadata.query("(BIRADS == 4.0)").sample(n=60)
    meta_5f = metadata.query("(BIRADS == 5.0)").sample(n=12)
    metadata = pd.concat([meta_4f, meta_5f], axis=0)

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
        except:
            print(sid)
            print(lat)
            print(label)
            raise
    print("Total # of missing files:", len(error_list))
    dir_meta = dir_meta.dropna()

    # print(error_list)
    dir_meta.Patient = dir_meta.Patient.astype("int")

    return dest_dir, dir_meta


dest_dir, dir_meta = label_dir(
    source_dir=FLAT_PATH, dest_dir=TARGET_PATH, metadata=meta_df
)
# Keep the last substring of TARGET_PATH after the last slash
dir_meta.to_csv(f"{BASE_DIR}/data/metadata/{TARGET_PATH.split('/')[-1]}.csv")

#%%

# ### THIS BLOCK IS OPTIONAL TO PROVIDE MORE NON OVERLAPPING DATA
# # Create new TARGET_PATH with the difference between the files in two folders
# def difference_files(TARGET_PATH=TARGET_PATH, BASE_DIR=BASE_DIR):
#     """Creates a new TARGET_PATH with the difference between the files in two folders

#     Returns:
#         path of target folder (str)
#     """
#     # Create diff directory inside data/reader_study_expanded
#     diff_dir = f"{TARGET_PATH}/diff"
#     os.makedirs(diff_dir)

#     d1 = os.listdir(f"{BASE_DIR}/data/reader_study/malignant")
#     d2 = os.listdir(f"{BASE_DIR}/data/reader_study_expanded/malignant")
#     d1 = [f"{BASE_DIR}/data/reader_study/malignant/{i}" for i in d1]
#     d2 = [f"{BASE_DIR}/data/reader_study_expanded/malignant/{i}" for i in d2]
#     d1 = set(d1)
#     d2 = set(d2)  # Keep only filenames in d1 and d2
#     d1 = [i.split("/")[-1] for i in d1]
#     d2 = [i.split("/")[-1] for i in d2]
#     # Get difference between d1 and d2
#     diff = list(set(d2) - set(d1))
#     # Copy files from d2 to diff_dir
#     for i in diff:
#         src = f"{BASE_DIR}/data/reader_study_expanded/malignant/{i}"
#         dest = f"{BASE_DIR}/data/reader_study_expanded/diff/{i}"
#         shutil.copyfile(src, dest)

#     return diff


# # Drop for dir_meta the files that are not in diff list
# diff = difference_files(TARGET_PATH=TARGET_PATH, BASE_DIR=BASE_DIR)
# # Remove .tiff from diff list
# diff = [i.split(".")[0] for i in diff]

# #%%
# # Split each element of diff into two lists diff_id and diff_laterality based on the last character
# diff_id = [i[:-1] for i in diff]
# diff_laterality = [i[-1] for i in diff]
# # Filter the rows of dir_meta that are in diff_id and diff_laterality for the column StudyID and Laterality
# dir_meta = dir_meta[
#     (dir_meta["StudyID"].isin(diff_id)) & (dir_meta["Laterality"].isin(diff_laterality))
# ]
# # Save as csv
# dir_meta.to_csv(f"{BASE_DIR}/data/metadata/reader_study_expanded.csv")

# %%
