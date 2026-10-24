# MultiWaveLab2

This work belongs to the Master's dissertation project conducted by Sofia P. Klautau at the Biomedical Engineering Institute at the Federal University of Santa Catarina (IEB-UFSC), Florianópolis, Brazil, under the guidance of Prof. Cesar R. Rodrigues.

This work was approved by the Ethics in Research with Humans Committee at UFSC, under the CAAE number: 84846024.9.0000.0121.

## Documentation

This fork was made to explore and present the feature extraction capability of this repo and to create new features to show as columns. This is better detailed in /new_features/README.md

## Reproducing the dissertation experiments

The complete set of experiments reported in the dissertation can be reproduced by running:

```bash
python sofias_dissertation_results.py
```

This script sequentially executes the processing and machine learning pipelines required to reproduce the dissertation experiments.

> **Note:** Full execution may take several hours, depending on the computer and available resources. The script is intended for complete reproducibility rather than as a quick example.

## Installing MultiWaveLab

### On Windows
Using Windows CMD, clone this repository. Then create a virtual environment with Python 3.10.13. Use pip to install the required packages from file ```requirements.txt```. We will be using two biomedical signal processing libraries: PyPPG and NeuroKit2. PyPPG has dependencies on some outdated packages, such as Numpy 1.x. To avoid conflicts with NeuroKit2, which is more actively maintained and has more frequent updates, we must install PyPPG with no dependencies. Therefore, after installing the requirements, run ```pip install pyPPG --no-deps```.  

### On Linux
Using Ubuntu Bash, first install Miniconda or Anaconda and then clone this repository. With Conda, the installation is the same as the one for Windows. Create a virtual environment with Python 3.10.13. Then use pip to install the required packages from file ```requirements.txt```. After installing the requirements, run ```pip install pyPPG --no-deps```.

##### Miniconda installation example
````bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
````

### MultiWaveLab full installation using Conda:

```bash
conda create -n pyppg_env python=3.10.13
conda activate pyppg_env
pip install -r requirements.txt
pip install pyPPG --no-deps
```
