import os
import shutil
from sys import meta_path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"
BASE_FILE = "metadata"
FLAT_PATH = f"{BASE_DIR}/data/flat_raw"
TARGET_PATH = f"{BASE_DIR}/data/reader_study_6"


meta_df = pd.read_csv(f"{BASE_DIR}/data/metadata/{BASE_FILE}.csv", index_col=0)
meta_df = meta_df[meta_df.BIRADS == 6.0]
meta_df = meta_df[meta_df["Final Class Benign/Malignant"] == "Malignant"]
# meta_df = meta_df[meta_df.Indication == "Screening"]
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
    # Subsample metadata
    metadata = metadata.sample(n=15)
    dir_meta = dir_meta

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
            dir_meta = pd.concat([metadata, pd.DataFrame(data_dic, index=[0])])

        except FileNotFoundError:
            error_list.append(src + "----" + dest)
        except:
            print(sid)
            print(lat)
            print(label)
            raise
    print("Total # of missing files:", len(error_list))
    # dir_meta = dir_meta.dropna()

    # print(error_list)
    # dir_meta.Patient = dir_meta.Patient.astype("int")

    return dest_dir, dir_meta


dest_dir, dir_meta = label_dir(
    source_dir=FLAT_PATH, dest_dir=TARGET_PATH, metadata=meta_df
)
