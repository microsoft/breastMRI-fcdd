#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load MRI data
data = pd.read_csv("/home/felipeoviedoperhavec/ssdshared/FH/data/metadata/metadata.csv")
data["MRIdate"] = pd.to_datetime(data["MRIdate"])
data = data.sort_values(by="MRIdate")
data = data.dropna(subset=["Final Class Benign/Malignant"])
data["StudyCode"] = data["StudyId"] + data["MRILaterality"]

# Load high risk positive patients
data_positives = pd.read_csv(
    "/home/felipeoviedoperhavec/ssdshared/FH/data/metadata/cancer_risk_n88.csv"
)
data_positives.columns = ["PatientId", "MRILaterality"]
data_positives["PatientId"] = data_positives["PatientId"].astype("str")
data_positives["Develops_Cancer"] = True

# Load diagnosis dataframe
data_diagnosis = pd.read_csv(
    "/home/felipeoviedoperhavec/ssdshared/FH/data/metadata/diagnosis_date.csv"
)
data_diagnosis.columns = [
    "PatientId",
    "MRILaterality",
    "DayMissing",
    "DIAGdate",
    "DIAGdate_full",
]
data_diagnosis["DIAGdate"] = pd.to_datetime(data_diagnosis["DIAGdate"])
data_diagnosis = data_diagnosis[
    ["PatientId", "MRILaterality", "DayMissing", "DIAGdate"]
]
data_diagnosis = data_diagnosis.dropna()
data_diagnosis.PatientId = data_diagnosis["PatientId"].astype("str")

# Positive diagnosis
data_pos = pd.merge(
    data_diagnosis, data_positives, how="right", on=["PatientId", "MRILaterality"]
)

data_pos = data_pos.drop(["DayMissing"], axis=1)
data_pos.PatientId = data_pos.PatientId.astype("int64")
# Export dataframe for risk cohort with Diagnosis date
data_pos.to_csv("/home/felipeoviedoperhavec/ssdshared/FH/data/metadata/risk_cohort.csv")


#%%

# Cohort analysis
def generate_prog(side):
    dpath = (
        data[(data["MRILaterality"] == side)]
        .groupby("PatientId")["Final Class Benign/Malignant"]
        .apply(list)
    )

    dind = (
        data[(data["MRILaterality"] == side)]
        .groupby("PatientId")["Indication"]
        .apply(list)
    )

    ddate = (
        data[(data["MRILaterality"] == side)]
        .groupby("PatientId")["MRIdate"]
        .apply(list)
    )
    dcode = (
        data[(data["MRILaterality"] == side)]
        .groupby("PatientId")["StudyCode"]
        .apply(list)
    )

    dprog = pd.merge(pd.merge(dpath, dind, on="PatientId"), dcode, on="PatientId")
    dprog = pd.merge(dprog, ddate, on="PatientId")
    # dprog.columns = [i + f"_{side}" for i in dprog.columns]
    # dprog = dprog.rename(columns={f"PatientId_{side}": "PatientId"})
    dprog["MRILaterality"] = side

    return dprog


# Merge right and left breasts data
prog_r = generate_prog("R")
prog_l = generate_prog("L")
prog = pd.concat([prog_r, prog_l], axis=0)
prog = prog.reset_index()

# Left join with risk indication dataframe
prog_diag = pd.merge(prog, data_pos, on=["PatientId", "MRILaterality"], how="left")

# # Featurize first, last dates + risk / no risk indication
prog_diag["First"] = prog_diag["MRIdate"].apply(lambda x: x[0])
prog_diag["Last"] = prog_diag["MRIdate"].apply(lambda x: x[-1])
prog_diag.Develops_Cancer.fillna(False, inplace=True)
prog_diag["Diff_First"] = prog_diag["DIAGdate"] - prog_diag["First"]
prog_diag["Diff_Last"] = prog_diag["DIAGdate"] - prog_diag["Last"]

# # Drop malignant at time of first MRI, may not be consistent with indication
# prog_diag["Has_Cancer"] = prog_diag["Final Class Benign/Malignant"].map(
#     lambda x: True if x[0] == "Malignant" else False
# )
# prog_diag[prog_diag["Develops_Cancer"] == True]

# %%


def f(x):
    y = x["Final Class Benign/Malignant"][::-1]
    try:
        ind = y.index("Malignant") + 1
    except ValueError:
        ind = 0
    ind = len(x["Final Class Benign/Malignant"]) - ind - 1
    study = x["StudyCode"][ind]
    return study


def g(x):
    y = x["Final Class Benign/Malignant"][::-1]
    try:
        ind = y.index("Malignant") + 1
    except ValueError:
        ind = 0
    ind = len(x["Final Class Benign/Malignant"]) - ind - 1
    date = x["MRIdate"][ind]
    return date


prog_diag["Last_Benign"] = prog_diag.apply(f, axis=1)
prog_diag["Last_Benign_Date"] = prog_diag.apply(g, axis=1)
prog_diag["Diff_Last"] = prog_diag["DIAGdate"] - prog_diag["Last_Benign_Date"]
prog_diag.to_csv(
    "/home/felipeoviedoperhavec/ssdshared/FH/data/metadata/risk_cohort_granular.csv"
)

# %%
