""" 
Preprocessing data pipeline.
Moves from original folder structures (BIRADs based) to flat bening/malignant split.
Split folders into train-test splits, one random and one grouped; and uses the final FCDD 
directory structure.

"""

import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

BASE_DIR = "/home/felipeoviedoperhavec/ssdprivate/FH"  # Adjust accordingly
metadata_df = pd.read_csv(f"{BASE_DIR}/data/metadata/metadata.csv", index_col=0)
BASE_PATH = f"{BASE_DIR}/BreastData_15-Dec-2018"
FLAT_PATH = f"{BASE_DIR}/data/flat_raw"  # Path for flattened data
# TARGET_PATH = "data/full_data" # legacy data folder
TARGET_PATH = f"{BASE_DIR}/data/flat_data"  # TFlat directory with labelled folders
FCDD_PATH = f"{BASE_DIR}/data/fccd_data"  # Path to FCDD
RISK_PATH = (
    f"{BASE_DIR}/data/metadata/risk_cohort_granular.csv"  # Path to labelled 5 year risk
)


def flat_dir(source_dir, dest_dir):
    """Flattens directory of raw data starting from BIRADS splits.

    Args:
        source (str): source path.
        dest (str): destination path.
    """
    # Check if dest_dir exists, if it does, delete it, then re-create folder structure
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    # Make directories
    os.makedirs(dest_dir)

    for dirpath, _, filenames in os.walk(source_dir, topdown=False):
        for filename in filenames:
            i = 0
            source = os.path.join(dirpath, filename)
            target = os.path.join(dest_dir, filename)

            while os.path.exists(target):
                "Appends addittional number of there is a file collosion issue."
                i += 1
                file_parts = os.path.splitext(os.path.basename(filename))

                target = os.path.join(
                    source,
                    file_parts[0] + "_" + str(i) + file_parts[1],
                )
                print(f"Collision in: {target}")

            shutil.copy(source, target)


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


def create_split(dir_df, n_splits=5, grouping=None, fold=None):
    """_summary_

    Args:
        dir_df (_type_): _description_
        n_splits (int, optional): _description_. Defaults to 5.
        grouping (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if fold is None:
        n_fold = 0  # Picks first fold
    if fold is not None:
        n_fold = fold  # Splits into the right folds

    if not grouping:
        kf = KFold(n_splits=n_splits, random_state=12, shuffle=True)
        train_ind, test_ind = list(kf.split(dir_df))[n_fold]  # Choose first fold

    if grouping:
        if grouping == "Study":
            groups = dir_df.StudyID
        elif grouping == "Patient":
            groups = dir_df.Patient
        kf = GroupKFold(n_splits=n_splits)
        train_ind, test_ind = list(kf.split(dir_df, groups=groups))[n_fold]

    train_df = dir_df.iloc[train_ind]
    test_df = dir_df.iloc[test_ind]

    return train_df, test_df


def create_fcdd_dir(
    dir_df,
    n_splits,
    grouping=None,
    dest_dir=FCDD_PATH,
    suffix=None,
    filter_case=None,
    fold=None,
):

    # Add suffix, for path consistency
    if suffix:
        dest_dir = dest_dir + f"_{suffix}"  # Suffix
    if fold is not None:
        dest_dir = dest_dir + f"_{fold}"  # Add fold when doing n-fold CV

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

    # Create split dataset with labels
    train_df, test_df = create_split(
        dir_df=dir_df, n_splits=n_splits, grouping=grouping, fold=fold
    )

    # Filter dataframe, we can still train with extreme cases
    if filter_case == "Known_Cancer":
        test_df = test_df[test_df.Indication != "Known Cancer"]
    elif filter_case == "Birads":
        test_df = test_df[test_df.BIRADS != 6.0]
    elif filter_case == "Both":
        test_df = test_df[test_df.Indication != "Known Cancer"]
        test_df = test_df[test_df.BIRADS != 6.0]

    # Copy files according to split
    copy_fcdd(train_df, dest_dir=dest_dir, df_type="train")
    copy_fcdd(test_df, dest_dir=dest_dir, df_type="test")

    # Save train-test split for future reference + grouping method
    if fold is not None:
        suffix = suffix + f"_{fold}"

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


def copy_risk_fcdd(df, dest_dir, df_type="train"):

    for pth, label, lat, sid in zip(df.Path, df.Risk_Label, df.Laterality, df.StudyID):
        try:
            if label == True:
                src = pth
                dest = (
                    f"{dest_dir}/custom/{df_type}/breast_img/anomalous/{sid}{lat}.tiff"
                )
                shutil.copyfile(src, dest)
            elif label == False:
                src = pth
                dest = f"{dest_dir}/custom/{df_type}/breast_img/normal/{sid}{lat}.tiff"
                shutil.copyfile(src, dest)

        except:
            print(f"Error in file {pth}")
            raise


def create_fcdd_risk(
    dir_df,
    n_splits,
    grouping=None,
    dest_dir=FCDD_PATH,
    suffix=None,
    filter_case=None,
    cumulative="Test",
    fold=None,
):

    # Add suffix, for path consistency
    if suffix:
        dest_dir = dest_dir + f"_{suffix}"  # Suffix
    if fold is not None:
        dest_dir = dest_dir + f"_{fold}"  # Add fold when doing n-fold CV

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

    # Load dataframe of risk individual (develop cancer post benign MRI)
    risk_df = pd.read_csv(
        f"{RISK_PATH}",
        index_col=0,
    )
    risk_df.DIAGdate = risk_df.DIAGdate.astype("datetime64[ns]")
    risk_df.rename(
        columns={"PatientId": "Patient", "MRILaterality": "Laterality"}, inplace=True
    )
    risk_df = risk_df[
        [
            "Patient",
            "Laterality",
            "DIAGdate",
            "Develops_Cancer",
            "First",
            "Last",
            "Last_Benign_Date",
            "Diff_First",
            "Diff_Last",
        ]
    ]

    # Format date in dir_df
    dir_df.MRIdate = dir_df.MRIdate.astype("datetime64[ns]")
    df = pd.merge(dir_df, risk_df, how="left", on=["Patient", "Laterality"])
    # # Registers if MRI happened after diagnosis date
    # df["After_Diag"] = df["MRIdate"] >= df["DIAGdate"]
    # # Label risk, if MRI Benign, develops cancer in breast and MRI happened before diagnosis
    df["Risk_Label"] = df.apply(label_risk, axis=1)
    # Filter out of range, high risk patients outside of 5-year period
    df = df[df["Risk_Label"] != "Out of Range"]

    # Create split dataset with labels
    train_df, test_df = create_split(
        dir_df=df, n_splits=n_splits, grouping=grouping, fold=fold
    )

    # Cumulative (inlcude 1 year risk into 5 or not - for testing)
    if (
        cumulative == "Test"
    ):  # Tests in cases that develops within 1-5 years cancer, but not before
        test_df = test_df[test_df["Risk_Label"] != "Malignant"]
    elif (
        cumulative == "Train-Test"
    ):  # Drop cases that develop cancer within 1 year from test and train. Lowest amount of data available, but cleanest -- no malignant cases in train or test
        test_df = test_df[test_df["Risk_Label"] != "Malignant"]
        test_df = test_df[test_df["BIRADS"] != 6.0]
        test_df = test_df[test_df["Indication"] != "Known Cancer"]
        train_df = train_df[train_df["Risk_Label"] != "Malignant"]  # New addition
        train_df = train_df[train_df["BIRADS"] != 6.0]  # New addition
        train_df = train_df[train_df["Indication"] != "Known Cancer"]  # New addition
    elif (
        cumulative == "Test-KC"
    ):  # Drop cases that develop cancer within 1 year from test, this will be risk of 0 to 5 years, cumulative
        test_df = test_df[test_df["Risk_Label"] != "Malignant"]
        test_df = test_df[test_df["BIRADS"] != 6.0]
        test_df = test_df[test_df["Indication"] != "Known Cancer"]

    # Filter training data according to case (might reduce noise)
    if filter_case == "First":  # Keep only first ocurrence
        train_df["Is_First"] = train_df["MRIdate"] == train_df["First"]
        train_df = train_df[train_df.Is_First is True]
    elif filter_case == "Last":  # Keep only last benign ocurrence
        train_df["Is_Last"] = train_df["MRIdate"] == train_df["Last_Benign_Date"]
        train_df = train_df[train_df.Is_Last is True]

    # Map to final labels
    mapper = {"Malignant": True, "High Risk": True, "Low Risk": False}
    train_df.replace({"Risk_Label": mapper}, inplace=True)
    test_df.replace({"Risk_Label": mapper}, inplace=True)

    # Copy files according to split
    copy_risk_fcdd(train_df, dest_dir=dest_dir, df_type="train")
    copy_risk_fcdd(test_df, dest_dir=dest_dir, df_type="test")

    # Save train-test split for future reference + grouping method
    if fold is not None:
        suffix = suffix + f"_{fold}"

    train_df.to_csv(
        f"{BASE_DIR}/data/split_logs/train-{grouping}-{filter_case}-{cumulative}-{suffix}.csv"
    )
    test_df.to_csv(
        f"{BASE_DIR}/data/split_logs/test-{grouping}-{filter_case}-{cumulative}-{suffix}.csv"
    )

    return df


def create_all_risk(
    dir_df,
    dest_dir=FCDD_PATH,
    suffix=None,
):
    # Puts all the risk cohort pictures (those that develop or not cancer) on a single folder, for validation and exploration
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

    # Load dataframe of risk individual (develop cancer post benign MRI)
    risk_df = pd.read_csv(
        f"{RISK_PATH}",
        index_col=0,
    )
    risk_df.DIAGdate = risk_df.DIAGdate.astype("datetime64[ns]")
    risk_df.rename(
        columns={"PatientId": "Patient", "MRILaterality": "Laterality"}, inplace=True
    )
    risk_df = risk_df[
        [
            "Patient",
            "Laterality",
            "DIAGdate",
            "Develops_Cancer",
            "First",
            "Last",
            "Last_Benign_Date",
            "Diff_First",
            "Diff_Last",
        ]
    ]

    # Format date in dir_df
    dir_df.MRIdate = dir_df.MRIdate.astype("datetime64[ns]")
    df = pd.merge(dir_df, risk_df, how="left", on=["Patient", "Laterality"])
    # # Registers if MRI happened after diagnosis date
    # df["After_Diag"] = df["MRIdate"] >= df["DIAGdate"]
    # # Label risk, if MRI Benign, develops cancer in breast and MRI happened before diagnosis
    df["Risk_Label"] = df.apply(label_risk, axis=1)
    # Filter out of range, high risk patients outside of 5-year period
    df = df[df["Risk_Label"] != "Out of Range"]
    # Create a new dataframe by filtering out cases with Risk_Label == "Low Risk"
    # df_copy = df[df["Risk_Label"] != "Low Risk"]
    df_copy = df.copy()

    # Map to final labels
    mapper = {"Malignant": True, "High Risk": True, "Low Risk": False}
    df_copy.replace({"Risk_Label": mapper}, inplace=True)

    # Copy files according to split
    copy_risk_fcdd(df_copy, dest_dir=dest_dir, df_type="test")
    df["Risk_Label_Mapped"] = df_copy["Risk_Label"]
    df.to_csv(f"{BASE_DIR}/data/split_logs/all_risk-{suffix}.csv")

    return df


def label_risk(x):
    if x["Label"] == "Benign" and x["Develops_Cancer"] is True:
        diff = x["DIAGdate"] - x["MRIdate"]
        # If diagnositc within ~5 years (negative and positive offsets account for anonymization):
        if np.timedelta64(-45, "D") < diff < np.timedelta64(1885, "D"):
            return "High Risk"
        return "Out of Range"
    elif x["Label"] == "Benign" and x["Develops_Cancer"] is False:
        return "Low Risk"
    elif x["Label"] == "Malignant":
        return "Malignant"
    else:
        return "Other"


if __name__ == "__main__":

    # Flatten directory
    flat_dir(source_dir=BASE_PATH, dest_dir=FLAT_PATH)

    # Split according to labels
    whole_dir, whole_df = label_dir(source_dir=FLAT_PATH, dest_dir=TARGET_PATH)

    # create_all_risk(whole_df, dest_dir=FCDD_PATH, suffix="all_risk_test")

    # # No grouping, case filtering
    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping=None,
    #     suffix="baseline",
    #     filter_case=None,
    # )

    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping=None,
    #     suffix="baseline_kc",
    #     filter_case="Known_Cancer",
    # )

    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping=None,
    #     suffix="baseline_br",
    #     filter_case="Birads",
    # )

    # # Grouping Patient + Case filtering
    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping="Study",
    #     suffix="study",
    #     filter_case=None,
    # )

    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping="Patient",
    #     suffix="patient",
    #     filter_case=None,
    # )

    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping="Study",
    #     suffix="study_kc",
    #     filter_case="Known_Cancer",
    # )

    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping="Patient",
    #     suffix="patient_kc",
    #     filter_case="Known_Cancer",
    # )

    # Creates Revised Task #1 - no CV
    # create_fcdd_dir(
    #     whole_df,
    #     n_splits=5,
    #     grouping="Patient",
    #     suffix="patient_kc_both",
    #     filter_case="Both",
    # )

    # Creates Revised Task #1 - 5-fold split
    # for i in range(0, 5):
    #     create_fcdd_dir(
    #         whole_df,
    #         n_splits=5,
    #         grouping="Patient",
    #         suffix="patient_kc_both_cv",
    #         filter_case="Both",
    #         fold=i,  # This takes care of folder names too.
    #     )

    # Creates Revised Task #0 - 5-fold split
    for i in range(0, 5):
        create_fcdd_dir(
            whole_df,
            n_splits=5,
            grouping="Patient",
            suffix="patient_task0_cv",
            filter_case=None,
            fold=i,  # This takes care of folder names too.
        )

    # # Creates Revised Task #0 - 5-fold split, filtering known cancer
    # for i in range(0, 5):
    #     create_fcdd_dir(
    #         whole_df,
    #         n_splits=5,
    #         grouping="Patient",
    #         suffix="patien_task0_cv_kc",
    #         filter_case="Known_Cancer",
    #         fold=i,  # This takes care of folder names too.
    #     )

# create_fcdd_dir(
#     whole_df,
#     n_splits=5,
#     grouping="Study",
#     suffix="study_br",
#     filter_case="Birads",
# )

# create_fcdd_dir(
#     whole_df,
#     n_splits=5,
#     grouping="Patient",
#     suffix="patient_br",
#     filter_case="Birads",
# )

## Risk labels (develops cancer after more than 1 year after benign), for all groupings
# check = create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping=None,
#     suffix="baseline_rk",
# )

# check = create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping="Patient",
#     suffix="patient_rk",
#     cumulative="Test-KC",
# )

# for i in range(0, 5):
#     check = create_fcdd_risk(
#         whole_df,
#         n_splits=5,
#         grouping="Patient",
#         suffix="patient_rk_cv_ck",
#         cumulative="Test-KC",
#         fold=i,
#     )

# check = create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping="Study",
#     suffix="study_rk",
# )

# create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping="Patient",
#     suffix="full_ctest_pt",
#     filter_case=None,
#     cumulative="Test",
# )

# create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping="Patient",
#     suffix="first_ctest_pt",
#     filter_case="First",
#     cumulative="Test",
# )

# create_fcdd_risk(
#     whole_df,
#     n_splits=5,
#     grouping="Patient",
#     suffix="last_ctest_tt",
#     filter_case="Last",
#     cumulative="Test",
# )
