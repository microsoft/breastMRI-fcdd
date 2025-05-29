# FCDD
Code for deep explainable anomaly detection, based on the Fully Convolutional Data Description (FCDD) model (Liznerski et al., 2021). Fork from repo: https://github.com/liznerski/fcdd

## Installation
It is recommended to use a Conda virtual environment to install FCDD.
Assuming you're in the `python` directory in the repository, install FCDD via pip:

    conda create --name fcdd python=3.9
    conda activate fcdd
    cd python
    pip install .

After installation, check that CUDA is detected by PyTorch:

    python -c "import torch; print(torch.cuda.is_available())"

## Usage

### Training

Let **dsdir** be your specified dataset directory (per default `../../data/datasets/`). 
Place your training data in `dsdir/custom/train/classX/` and your test data in `dsdir/custom/test/classX/`, with classX being one of the class folders (they can have arbitrary names, but need to be consistent for training and testing).

Each class requires a separate set of nominal and anomalous test samples.
Place the corresponding images in `dsdir/custom/test/classX/normal/`, `dsdir/custom/test/classX/anomalous/`, `dsdir/custom/train/classX/normal/`, `sdir/custom/train/classX/anomalous/`, and run:

    python runners/run_custom.py --supervise-mode other
    
In general, you can adapt most training parameters using the program's arguments (see `python runners/run_custom.py --help`).

## Training examples

- For Task 1 (Balanced detection), an example fold (train/test split grouped by patient) is located in "/home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_0"
- For Task 2 (Imbalanced detection) an example fold is located in "/home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_kc_both_cv_0"

For training, the parameters are the same in both cases, and only the directory differs.

For $FCDD$, assuming we are focusing on Task 1, the training command run from the 'python/fcdd' directory is as follows:

    python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_0 --net FCDD_CNN224_VGG_NOPT --workers 4 --it 1 --epochs 1 --batch-size 32 --blur-heatmaps --objective fcdd --logdir-suffix task0_fcdd --gpu 0

Similarly for $BCE$ (Binary Cross Entropy), our usual baseline, we have:

    python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_0 --net VGG_BCE_CROP --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective bce --logdir-suffix task0_bce --gpu 0

For $HSC$, the Hypersphere Classification loss, our non-explainable anomaly detection model we have:

    python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_0 --net CNN224_CROP --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective hsc --logdir-suffix task0_hsc --gpu 0

Finally, we have the model variation $FCDD_Symmetric$. This model is a variation of $FCDD$ that uses a reference image for each training image. The mapping between the actual image and the reference image is included in the data directory in the .csv files 'train_ref.csv' and 'test_ref.csv'. Here, each breast has as reference the contralateral breast when possible (contra lateral breast is non malignant). Otherwise, the reference breast is randomly selected from a benign breast in the dataset without data leakage.

In this training command is:

    python run_scans_refs.py --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_0 --net FCDD_REF_CNN224_VGG_NOPT --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective fcddrefs --logdir-suffix task0_fcdd_ref --gpu 0


## Explanation of the Log Data after Training

A training run saves the achieved scores, metrics, plots, snapshots, and heatmaps in a given log directory in "../data/results" (created during training if it does not exist yet).
Each log directory contains a separate subdirectory for each class that is trained to be nominal. 
These subdirectories are named "normal_x", where x is the class number. 
The class subdirectories again contain a subdirectory for each random seed.
These are named "it_x" for x being the iteration number (random seed). 
Inside the seed subdirectories all actual log data can be found. 
Additionally, summarized plots will be created for the class subdirectories and the root log directory. 
For instance, a plot containing ROC curves for each class (averaged over all seeds) can be found in the root log directory.

Visualization for 2 classes and 2 random seeds: 

    ./log_directory 
    ./log_directory/normal_0 
    ./log_directory/normal_0/it_0 
    ./log_directory/normal_0/it_1
    ./log_directory/normal_1 
    ./log_directory/normal_1/it_0 
    ./log_directory/normal_1/it_1 
    ...
    
Note that the leaf nodes, i.e. the iteration subdirectories, contain completely separate training results and have no impact on each other. 

The actual log data consists of: 
- **config.txt**: A file containing all the arguments of the training.
- **ds_preview.png**: A preview of the dataset, i.e. some random nominal and anomalous samples from the training set. Includes augmentation and data preprocessing. Also shows corresponding ground-truth maps, if such are available. 
- **err.pdf**: A plot of the loss over time.
- **err_anomalous.pdf**: A plot of the loss for anomalous samples only.
- **err_normal.pdf**: A plot of the loss for nominal samples only.
- **heatmaps_paper_x_lbly.png**: An image of test heatmaps, test inputs, and ground-truth maps. One image for each x-y combination. x is the normalization used, either local (each heatmap is normalized w.r.t. itself only) or semi_global (each heatmap is normalized w.r.t. to all heatmaps in the image). y is the label, i.e. either 1 or 0 (per default 1 is for anomalies).
- **heatmaps_global.png**: An image of the first test heatmaps and inputs found in the dataset. The first row shows nominal samples, the second-row anomalous samples. The third row shows the ten most nominal rated nominal samples on the left and the ten most anomalous rated nominal samples on the right. The fourth row shows the ten most nominal rated anomalies on the left and the ten most anomalous rated anomalies on the right. Note that changing the nominal label to 1 flips this. 
- **train_heatmaps_global.png**: Like above, but for training samples.
- **history.json**: A file containing metrics in text form.
- **log.txt**: A file containing some logged text lines, like the average AUC value achieved and the duration of the training.
- **print.log**: A file that contains all text that has been printed on the console using the Logger.
- **roc.json**: ROC values saved as text (not very readable for humans). 
- **roc_curve.pdf**: ROC for detection performance.
- **gtmap_roc_curve.pdf**: ROC for explanation performance. Only for MVTec-AD (or datasets with ground-truth maps). 
- **snapshot.pt**: Snapshot of the training state and model parameters.
- **src.tar.gz**: Snapshot of the complete source code at the moment of the training start time. 
- **tims**: A directory containing raw tensors heatmaps (not readable for humans). 


## Performing Inference and Generating Heatmaps

Once the model is trained, you can easily perform inference using the unified prediction script. The script automatically loads the appropriate model snapshot and generates predictions along with heatmaps.

### Basic Usage

To run predictions with default settings:

    python runners/predict_custom.py --model fcdd --task 1

### Available Models and Tasks

The script supports four different models:
- `fcdd`: Fully Convolutional Data Description
- `bce`: Binary Cross Entropy baseline
- `hsc`: Hypersphere Classification
- `fcdd_ref`: FCDD with reference images

And two tasks:
- `task 1`: Balanced detection
- `task 2`: Imbalanced detection

### Command Examples

For FCDD on Task 1 (Balanced detection):

    python runners/predict_custom.py --model fcdd --task 1

For BCE on Task 2 (Imbalanced detection):

    python runners/predict_custom.py --model bce --task 2

For HSC with custom paths:

    python runners/predict_custom.py --model hsc --task 1 --snapshot_path /path/to/model/snapshot --output_dir /path/to/output

### Command Arguments

- `--model`: Model type (`fcdd`, `bce`, `hsc`, `fcdd_ref`)
- `--task`: Task number (1 or 2)
- `--snapshot_path`: Custom path to model snapshot directory (optional)
- `--output_dir`: Custom output directory for results (optional)
- `--device`: GPU device number (default: 0)

### Output

The script generates:
- **predictions_results.json**: Contains model predictions, scores, labels, and metrics (ROC AUC, PR AUC)
- **Heatmap images**: Anomaly heatmaps for each image in the test set, saved in the output directory. Heatmaps are locally and globally normalized.

The predictions are automatically saved and the file paths are correctly mapped to handle any reordering that occurs during data processing.