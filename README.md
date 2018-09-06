Code for TAC 2016 KBP Event Nugget Detection
========================================================

## About
This repository contains the code used for the paper:

Mihaylov and Frank (2016):
[AIPHES-HD system at TAC KBP 2016: Neural Event Trigger Span Detection and Event Type and Realis Disambiguation with Word Embeddingss. Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task.](https://tac.nist.gov/publications/2016/participant.papers/TAC2016.aipheshd_t16.proceedings.pdf)

```
@inproceedings{mihaylovfrank:2016,
  author = {Todor Mihaylov and Anette Frank},
  title = {{AIPHES-HD system at TAC KBP 2016: Neural Event Trigger Span Detection and Event Type and Realis Disambiguation with Word Embeddingss}},
  year = {2016},
  booktitle = {In Proceedings of the TAC Knowledge Base Population (KBP) 2016.},
  url = {https://tac.nist.gov/publications/2016/participant.papers/TAC2016.aipheshd_t16.proceedings.pdf},
}
```

### Main TAC Event Track
- http://www.nist.gov/tac/2016/KBP/Event/index.html

### Event nuget detection and coreference tasks
- http://cairo.lti.cs.cmu.edu/kbp/2016/event/index
- Rich ERE Annotation Guidelines Overview V4.2 http://www.nist.gov/tac/2016/KBP/guidelines/summary_rich_ere_v4.2.pdf
- Task description - http://cairo.lti.cs.cmu.edu/kbp/2016/event/Event_Mention_Detection_and_Coreference-2016-v1.pdf

## Setup environment

### Create virtual environment

```bash
virtualenv venv
```

Activate the environment:
```bash
cd venv
source bin/activate
```

### Install TensorFlow with CPU in virtualenv
Activate the environment
```bash
# activate the environment
sudo pip install --upgrade virtualenv

# Ubuntu/Linux 64-bit, CPU only, Python 2.7
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

# Install tensorflow
pip install --upgrade $TF_BINARY_URL
```

### Install TensorFlow with GPU in virtualenv
Activate the environment
```bash
# Login to cluster with GPU units
# HD ICL - https://wiki.cl.uni-heidelberg.de/foswiki/bin/view/Main/FaQ/Tutorials/GridengineTutorial#GPU_nodes
ssh cluster
# login to the GPU server
qlogin -l has_gpu=YES,h_rt=3600 -q gpu_short.q -now n

# new login

# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin/:$PATH

# activate the environment
sudo pip install --upgrade virtualenv

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

# Install tensorflow
pip install --upgrade $TF_BINARY_URL

# test if tensorflow works
python
import TensorFlow as tf # If this does not fail you are okay!
```

### v2 - GPU Install TensorFlow with GPU in virtualenv
Activate the environment
```bash
# Login to cluster with GPU units
# HD ICL - https://wiki.cl.uni-heidelberg.de/foswiki/bin/view/Main/FaQ/Tutorials/GridengineTutorial#Quickstart
ssh cluster
# login to the GPU server gpu3 - GTX 1080, 8GB
qlogin -l has_gpu=YES,hostname=gpu03 -q gpu_long.q # get a login on gpu02 in gpu_long.q



# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin/:$PATH

# activate the environment
sudo pip install --upgrade virtualenv

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc1-cp27-none-linux_x86_64.whl

# Install tensorflow
pip install --upgrade $TF_BINARY_URL

# test if tensorflow works
python
import TensorFlow as tf # If this does not fail you are okay!
```

### Install everything from requirements.txt
pip install -r requirements.txt

### Install Jupyter (for experiments)
sudo pip install jupyter

### Download CoreNLP
\corenlp\download_core_nlp.sh

### Install PyWrapper https://github.com/brendano/stanford_corenlp_pywrapper
git clone https://github.com/brendano/stanford_corenlp_pywrapper
cd stanford_corenlp_pywrapper
pip install .

### Install Java 8
https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04

sudo apt-get update
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo update-alternatives --config java
sudo nano /etc/environment
source /etc/environment

## Data and Resources
Obtain the training and eval data from LDC:
### English
Train Eval 2014/15: DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015
Eval 2016:

## How to run

### Download and preprocess the data data

1. Download the LDC data (see data/README.md)
2. Preprocess the data:
The input format is parsed using CORENLP and stored in json files that are used for training and evaluation.
* Set the paths in parse_data_2016.sh
* bash parse_data_2016.sh

### Train and eval new model

To train a model with the enhanced BiLSTM:
```bash
# modify paths in the script below and run
bash scripts/ex6_detect_events_v6_bilstm_v6_posdep_stacked_depattention_run_server_et18_node0_all.sh
```

To evaluate:
```bash
# modify paths in the script below and run
bash scripts/ex6_detect_events_v6_bilstm_v6_posdep_stacked_depattention_run_server_et18_node_test.sh
```