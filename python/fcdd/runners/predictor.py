""" Runs FCDD and compute metric on test from snapshot
"""
#%%
from operator import truediv

import numpy as np
import pandas as pd
import torch
from fcdd.datasets.image_folder import ADImageFolderDataset
from fcdd.models.bce_224 import VGG_BCE, VGG_BCE_CROP
from fcdd.models.shallow_cnn_224 import CNN224_CROP
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_NOPT
from fcdd.training.bce import BCETrainer
from fcdd.training.fcdd import FCDDTrainer
from fcdd.training.fcdd_refs import FCDDRefsTrainer
from fcdd.training.hsc import HSCTrainer
from fcdd.util.logging import Logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from typing import List, Tuple
from torch import Tensor
from torch.utils.data.dataset import Dataset
import os
from fcdd.training.fcdd_refs import FCDDRefsTrainer
from fcdd.models.fcdd_ref_cnn_224 import FCDD_REF_CNN224_VGG_NOPT
from fcdd.datasets.image_folder_refs import ADImageRefDataset


def reorder(
    labels: List[int],
    loss: Tensor,
    anomaly_scores: Tensor,
    imgs: Tensor,
    outputs: Tensor,
    gtmaps: Tensor,
    grads: Tensor,
    refs: Tensor,
    ds: Dataset = None,
) -> Tuple[List[int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """returns all inputs in an identical new order if the dataset offers a predefined (random) order"""
    if ds is not None and hasattr(ds, "fixed_random_order"):
        assert (
            gtmaps is None
        ), "original gtmaps loaded in score do not know order! Hence reordering is not allowed for GT datasets"
        o = ds.fixed_random_order
        labels = (
            labels[o]
            if isinstance(labels, (Tensor, np.ndarray))
            else np.asarray(labels)[o].tolist()
        )
        loss, anomaly_scores, imgs = loss[o], anomaly_scores[o], imgs[o]
        outputs, gtmaps = outputs[o], gtmaps
        grads = grads[o] if grads is not None else None
        refs = refs[o] if refs is not None else None
    return labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, refs


# Load config.txt file from results_path and get everything after { as a dictionary
def load_config(results_path):
    config = {}
    with open(results_path + "config.txt", "r") as f:
        for line in f:
            if line.startswith("{"):
                # Remove the first and last characters
                line = line[1:-1]
                line = line.split(",")
                line = [i.strip() for i in line]
                line = [i.split(":") for i in line]
                line = [[i[0].strip(), i[1].strip()] for i in line]
                # Convert to dictionary
                config = {i[0]: i[1] for i in line}
                # Correct double quotes
                config = {i[1:-1]: config[i] for i in config}
                config = {i: config[i].replace('"', "") for i in config}

    # Replace "ssdshared" with "ssdprivate" in config["datadir"] for legacy paths
    # config["datadir"] = config["datadir"].replace("ssdshared", "ssdprivate")
    # config["gpu"] = 2

    return config


def load_model(config: dict, logger: Logger, device: int = 1):
    """Create trainer from config file"""
    # Pick the architecture that was used for the snapshot
    if config["net"] == "FCDD_CNN224_VGG_NOPT":
        net = FCDD_CNN224_VGG_NOPT((3, 224, 224), bias=True).cuda(device=device)
        # Load data and make predictions
        trainer = FCDDTrainer(
            net,
            None,
            None,
            (None, None, None),
            logger,
            config["objective"],
            int(config["gauss_std"]),
            float(config["quantile"]),
            64,
            blur_heatmaps=bool(config["blur_heatmaps"]),
            device=device,
        )
    elif config["net"] == "VGG_BCE_CROP":
        net = VGG_BCE_CROP((3, 224, 224), bias=True).cuda(device=device)
        # Load data and make predictions
        trainer = BCETrainer(
            net,
            None,
            None,
            (None, None, None),
            logger,
            config["objective"],
            int(config["gauss_std"]),
            float(config["quantile"]),
            64,
            blur_heatmaps=bool(config["blur_heatmaps"]),
            device=device,
        )
    elif config["net"] == "VGG_BCE":
        net = VGG_BCE((3, 224, 224), bias=True).cuda(device=device)
        # Load data and make predictions
        trainer = BCETrainer(
            net,
            None,
            None,
            (None, None, None),
            logger,
            config["objective"],
            int(config["gauss_std"]),
            float(config["quantile"]),
            64,
            blur_heatmaps=bool(config["blur_heatmaps"]),
            device=device,
        )

    elif config["net"] == "CNN224_CROP":
        net = CNN224_CROP((3, 224, 224), bias=True).cuda(device=device)
        # Load data and make predictions
        trainer = HSCTrainer(
            net,
            None,
            None,
            (None, None, None),
            logger,
            config["objective"],
            int(config["gauss_std"]),
            float(config["quantile"]),
            64,
            blur_heatmaps=bool(config["blur_heatmaps"]),
            device=device,
        )
    else:
        raise NotImplementedError("Model {} is not defined yet.".format(config["net"]))

    return trainer


def load_model_ref(config: dict, logger: Logger, device: int = 4):
    """Create trainer from config file"""
    net = FCDD_REF_CNN224_VGG_NOPT((3, 224, 224), bias=True).cuda(device=device)
    # Load data and make predictions
    trainer = FCDDRefsTrainer(
        net,
        None,
        None,
        (None, None, None),
        logger,
        config["objective"],
        int(config["gauss_std"]),
        float(config["quantile"]),
        64,
        blur_heatmaps=bool(config["blur_heatmaps"]),
        device=device,
    )

    return trainer


def predict_and_evaluate(
    results_path: str,
    log_path: str,
    generate_heatmaps: bool = False,
    device: int = 1,
    on_train: bool = False,
    data_dir_path: str = None,
    no_trainer: bool = False,
):

    config = load_config(results_path)
    logger = Logger(log_path)

    if data_dir_path is not None:
        config["datadir"] = data_dir_path

    # Define Dataset
    ds = ADImageFolderDataset(
        root=config["datadir"],
        normal_class=int(config["normal_class"]),
        preproc=config["preproc"],
        supervise_mode=config["supervise_mode"],
        noise_mode=config["noise_mode"],
        online_supervision=truediv,
        oe_limit=np.Infinity,
        logger=logger,
        nominal_label=int(config["nominal_label"]),
    )

    # Make predictions for test set unless specified otherwise
    if not on_train:
        d_test = ds.test_set
    if on_train:
        d_test = ds.train_set

    trainer = load_model(config, logger, device)
    trainer.load(results_path + "snapshot.pt")
    trainer.net.eval()

    # Make predictions
    all_fnames = [i[0].split("/")[-1].split(".")[0] for i in d_test.dataset.imgs]
    all_paths = [i[0] for i in d_test.dataset.imgs]
    all_indices = d_test.indices

    # Make predictions
    loader = DataLoader(d_test, batch_size=1, num_workers=8)

    all_anomaly_scores, all_inputs, all_labels, all_upsampled = [], [], [], []
    for inputs, labels in loader:
        inputs = inputs.cuda(device=device)
        with torch.no_grad():
            outputs = trainer.net(inputs)
            anomaly_scores = trainer.anomaly_score(
                trainer.loss(outputs, inputs, labels, reduce="none")
            )
            if config["net"] != "CNN224_CROP" and config["net"] != "VGG_BCE_CROP":
                upsampled_scores = trainer.net.receptive_upsample(
                    anomaly_scores,
                    reception=True,
                    std=int(config["gauss_std"]),
                    cpu=False,
                )  # Receptive upsample of scores for heatmap generation
            else:
                upsampled_scores = None
            all_anomaly_scores.append(anomaly_scores.cpu())
            if config["net"] != "CNN224_CROP" and config["net"] != "VGG_BCE_CROP":
                all_upsampled.append(upsampled_scores.cpu())
            else:
                all_upsampled.append(0)
            all_inputs.append(inputs.cpu())
            all_labels.append(labels)
    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)

    # Compute all anomaly scores
    if config["net"] != "CNN224_CROP" and config["net"] != "VGG_BCE_CROP":
        all_upsampled = torch.cat(all_upsampled)
        all_anomaly_scores = torch.cat(all_anomaly_scores)
    else:
        all_upsampled = None
        all_anomaly_scores = [t.unsqueeze(0) for t in all_anomaly_scores]
        all_anomaly_scores = torch.cat(all_anomaly_scores)

    all_scores = trainer.reduce_ascore(all_anomaly_scores)

    # Compute ROC_AUC score and PR_AUC score
    roc_auc = roc_auc_score(all_labels, all_scores)
    pr_auc = average_precision_score(all_labels, all_scores)
    print(f"ROC_AUC: {roc_auc:.4f} - PR_AUC: {pr_auc:.4f}")
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    # Save results to a json file
    results = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "all_scores": all_scores.tolist(),
        "all_anomaly_scores": all_anomaly_scores,
        "all_labels": all_labels.tolist(),
        "all_fnames": all_fnames,
        "all_paths": all_paths,
        "all_indices": all_indices,
        "all_upsampled": all_upsampled,
        "all_images": all_inputs,
    }

    if no_trainer:
        return results

    return results, trainer


def predict_and_evaluate_ref(
    results_path: str,
    log_path: str,
    generate_heatmaps: bool = False,
    device: int = 1,
    on_train: bool = False,
    data_dir_path: str = None,
    no_trainer: bool = False,
):

    config = load_config(results_path)
    logger = Logger(log_path)

    # Define Dataset
    ds = ADImageRefDataset(
        root=config["datadir"],
        normal_class=int(config["normal_class"]),
        preproc=config["preproc"],
        supervise_mode=config["supervise_mode"],
        noise_mode=config["noise_mode"],
        online_supervision=truediv,
        oe_limit=np.Infinity,
        logger=logger,
        nominal_label=int(config["nominal_label"]),
    )

    # Make predictions for test set unless specified otherwise
    if not on_train:
        d_test = ds.test_set
    if on_train:
        d_test = ds.train_set

    trainer = load_model_ref(config, logger, device)
    trainer.load(results_path + "snapshot.pt")
    trainer.net.eval()

    all_fnames = d_test.ref_df["Actual"]
    # For each element in all_fnames, get the filename only removing the path and .tiff
    all_fnames = [i.split("/")[-1].split(".")[0] for i in all_fnames]
    all_paths = d_test.ref_df["Actual"].to_list()
    all_indices = d_test.ref_df.index.to_list()

    # Make predictions
    loader = DataLoader(d_test, batch_size=1, num_workers=8)

    all_anomaly_scores, all_inputs, all_labels, all_upsampled = [], [], [], []
    for inputs, labels, refs in loader:
        inputs = inputs.cuda(device=device)
        refs = refs.cuda(device=device)

        with torch.no_grad():
            outputs, outputs_ref = trainer.net(inputs, refs)
            anomaly_scores = trainer.anomaly_score(
                trainer.loss(outs=outputs, ins=inputs, labels=labels, refs=outputs_ref, reduce="none")
            )
            upsampled_scores = trainer.net.receptive_upsample(
                anomaly_scores,
                reception=True,
                std=int(config["gauss_std"]),
                cpu=False,
            )
            all_anomaly_scores.append(anomaly_scores.cpu())
            all_upsampled.append(upsampled_scores.cpu())
            all_inputs.append(inputs.cpu())
            all_labels.append(labels)

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    all_upsampled = torch.cat(all_upsampled)
    all_anomaly_scores = torch.cat(all_anomaly_scores)
    all_scores = trainer.reduce_ascore(all_anomaly_scores)

    # Compute ROC_AUC score and PR_AUC score
    roc_auc = roc_auc_score(all_labels, all_scores)
    pr_auc = average_precision_score(all_labels, all_scores)
    print(f"ROC_AUC: {roc_auc:.4f} - PR_AUC: {pr_auc:.4f}")
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    # Save results to a json file
    results = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "all_scores": all_scores.tolist(),
        "all_anomaly_scores": all_anomaly_scores,
        "all_labels": all_labels.tolist(),
        "all_fnames": all_fnames,
        "all_paths": all_paths,
        "all_indices": all_indices,
        "all_upsampled": all_upsampled,
        "all_images": all_inputs,
    }
    if no_trainer:
        del trainer, inputs, labels, refs, outputs, outputs_ref, anomaly_scores, upsampled_scores
        return results

    del inputs, labels, refs, outputs, outputs_ref, anomaly_scores, upsampled_scores
    return results, trainer


def predict_and_evaluate_bce(
    results_path,
    log_path,
    generate_heatmaps=False,
    device=1,
    on_train=False,
    data_dir_path: str = None,
    no_trainer: bool = False,
):

    config = load_config(results_path)
    logger = Logger(log_path)

    if data_dir_path is not None:
        config["datadir"] = data_dir_path

    # Define Dataset
    ds = ADImageFolderDataset(
        root=config["datadir"],
        normal_class=int(config["normal_class"]),
        preproc=config["preproc"],
        supervise_mode=config["supervise_mode"],
        noise_mode=config["noise_mode"],
        online_supervision=truediv,
        oe_limit=np.Infinity,
        logger=logger,
        nominal_label=int(config["nominal_label"]),
    )

    # Make predictions for test set unless specified otherwise
    if not on_train:
        d_test = ds.test_set
    if on_train:
        d_test = ds.train_set

    trainer = load_model(config, logger, device)
    trainer.load(results_path + "snapshot.pt")

    # Send trainer net to device 1 using .to
    trainer.net.to(device).eval()
    #trainer.net.eval()

    # # Make predictions
    all_fnames = [i[0].split("/")[-1].split(".")[0] for i in d_test.dataset.imgs]
    all_paths = [i[0] for i in d_test.dataset.imgs]
    all_indices = d_test.indices

    # Make predictions
    loader = DataLoader(d_test, batch_size=16, num_workers=8)

    all_anomaly_scores, all_inputs, all_labels = [], [], []

    (
        labels,
        loss,
        anomaly_scores,
        imgs,
        outputs,
        gtmaps,
        grads,
        refs,
    ) = trainer._gather_data(
        loader,
    )
    labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, refs = reorder(
        labels,
        loss,
        anomaly_scores,
        imgs,
        outputs,
        gtmaps,
        grads,
        refs,
        ds=loader.dataset,
    )

    # # # Compute ROC_AUC score and PR_AUC score
    # roc_auc = roc_auc_score(labels, anomaly_scores)
    # pr_auc = average_precision_score(labels, anomaly_scores)
    # print(f"ROC_AUC: {roc_auc:.4f} - PR_AUC: {pr_auc:.4f}")
    # # Compute precision-recall curve
    # precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    # # Compute ROC curve
    # fpr, tpr, _ = roc_curve(labels, anomaly_scores)

    # Save results to a json file
    results = {
        # "roc_auc": roc_auc,
        # "pr_auc": pr_auc,
        # "precision": precision.tolist(),
        # "recall": recall.tolist(),
        # "fpr": fpr.tolist(),
        # "tpr": tpr.tolist(),
        "all_scores": anomaly_scores.tolist(),
        "all_anomaly_scores": outputs.tolist(),
        "all_labels": labels,
        "all_fnames": all_fnames,
        "all_paths": all_paths,
        "all_indices": all_indices,
        "all_images": imgs,
        "all_grads": grads,
    }
    if no_trainer:
        return results

    return results, trainer


def predict_and_evaluate_hsc(
    results_path,
    log_path,
    generate_heatmaps=False,
    device=1,
    on_train=False,
    data_dir_path: str = None,
    no_trainer: bool = False,
):

    config = load_config(results_path)
    logger = Logger(log_path)

    if data_dir_path is not None:
        config["datadir"] = data_dir_path

    # Define Dataset
    ds = ADImageFolderDataset(
        root=config["datadir"],
        normal_class=int(config["normal_class"]),
        preproc=config["preproc"],
        supervise_mode=config["supervise_mode"],
        noise_mode=config["noise_mode"],
        online_supervision=truediv,
        oe_limit=np.Infinity,
        logger=logger,
        nominal_label=int(config["nominal_label"]),
    )

    # Make predictions for test set unless specified otherwise
    if not on_train:
        d_test = ds.test_set
    if on_train:
        d_test = ds.train_set

    trainer = load_model(config, logger, device)
    trainer.load(results_path + "snapshot.pt")

    # Send trainer net to device 1 using .to
    # trainer.net.to(device).eval()
    trainer.net.eval()

    # # Make predictions
    all_fnames = [i[0].split("/")[-1].split(".")[0] for i in d_test.dataset.imgs]
    all_paths = [i[0] for i in d_test.dataset.imgs]
    all_indices = d_test.indices

    # Make predictions
    loader = DataLoader(d_test, batch_size=16, num_workers=8)
    all_anomaly_scores, all_inputs, all_labels = [], [], []

    (
        labels,
        loss,
        anomaly_scores,
        imgs,
        outputs,
        gtmaps,
        grads,
        refs,
    ) = trainer._gather_data(
        loader,
    )
    labels, loss, anomaly_scores, imgs, outputs, gtmaps, grads, refs = reorder(
        labels,
        loss,
        anomaly_scores,
        imgs,
        outputs,
        gtmaps,
        grads,
        refs,
        ds=loader.dataset,
    )
    # all_scores = trainer.reduce_ascore(all_anomaly_scores)

    # # Compute ROC_AUC score and PR_AUC score
    roc_auc = roc_auc_score(labels, anomaly_scores)
    pr_auc = average_precision_score(labels, anomaly_scores)
    print(f"ROC_AUC: {roc_auc:.4f} - PR_AUC: {pr_auc:.4f}")
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)

    # Save results to a json file
    results = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "all_scores": anomaly_scores.tolist(),
        "all_anomaly_scores": outputs.tolist(),
        "all_labels": labels,
        "all_fnames": all_fnames,
        "all_paths": all_paths,
        "all_indices": all_indices,
        "all_images": imgs,
        "all_grads": grads,
    }
    if no_trainer:
        return results

    return results, trainer

# FCDD usage
# results_path = "/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/fcdd_20220826232004fcdd_task1_cv_0_custom_/normal_0/it_0/"
# log_path = "../../../data/results/predicted2/"
# results = predict_and_evaluate(results_path=results_path, log_path=log_path)

# BCE usage
# results_path = "/home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/data/results/fcdd_20220922210714bce_task2_cv_0_crop_custom_/normal_0/it_0/"
# log_path = "../../../data/results/predicted2/"
# results = predict_and_evaluate_bce(results_path=results_path, log_path=log_path)