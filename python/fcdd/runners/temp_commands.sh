{"datadir": "/home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation", "objective": "fcdd", "batch_size": 32, "epochs": 200, "workers": 6, "learning_rate": 0.001, "weight_decay": 1e-06, "optimizer_type": "sgd", "scheduler_type": "lambda", "lr_sched_param": [0.985], "load": null, "dataset": "custom", "net": "FCDD_CNN224_VGG_NOPT", "preproc": "aug1", "acc_batches": 1, "bias": true, "cuda": true, "supervise_mode": "other", "noise_mode": "imagenet22k", "oe_limit": Infinity, "online_supervision": true, "nominal_label": 0, "blur_heatmaps": true, "gauss_std": 10, "quantile": 0.97, "resdown": 64, "gpu": "0", "normal_class": 0, "logdir": "../../data/results/fcdd_{t}fcdd_explanation_custom_/normal_0/it_0", "log_start_time": 1678832531, "viz_ids": []}


# Run FCDD with explanation on the Custom dataset
python runners/run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation --workers 6 --it 3 --objective fcdd -n FCDD_CNN224_VGG_NOPT --epochs 200 --batch-size 32 --blur-heatmaps --logdir-suffix fcdd_explanation

# Run FCDD with explanation on the Ref dataset
python runners/run_scans_refs.py --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fcdd_data_explanation --workers 6 --it 3 --objective fcddrefs -n FCDD_REF_CNN224_VGG_NOPT --epochs 200 --batch-size 32 --blur-heatmaps --logdir-suffix fcdd_explanation_ref --gpu 6

# Run training using gt mpas as reference
