# FCDD
Code for deep explainable anomaly detection, based on the Fully Convolutional Data Description (FCDD) model (Liznerski et al., 2021).

## Installation
It is recommended to use a virtual environment to install FCDD.
Assuming you're in the `python` directory, install FCDD via pip:

    conda create --name fcdd python=3.9
    conda activate fcdd
    cd python
    pip install .

After installation, check that CUDA is detected by PyTorch:

    python -c "import torch; print(torch.cuda.is_available())"

## Usage

### Training