# NYCU Computer Vision with Deep learning Spring 2026 HW2
Student Id: 313540041
Name: Ruy Calderon

## Introduction
This repo contains the digit detector for the Spring 2026 "Special Topics in Computer Vision using Deep Learning" course.

The model was written in python using the pytorch computer vision module, along with the required dependencies.

The project is run in a virtual environment managed by the conda deployment, and is run on the mac osx operating system

The model classifier is ResNet, with a DETR-style transformer detection head. 

The designated training, validation, and testing datasets are provided as part of the homework assignment.

## Environment Setup
Describe how to create the environment and install dependencies.

```bash
# Create and activate environment
conda create -n myenv python=3.10
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

# Project Overview

The python file `hw2.py` contains all the code required to train the classifier.

There are four possible command line arguments:

1. mode
2. root_dir
3. annotation_path
4. model_path

## Arguments


### mode
Select which mode you'd like to run in. There are two options:

1. `train` — Training  
2. `infer` — Inference  

### root_dir
This is the project root directory, it should contain a data path with a standard coco-style directory path in addition to a src directory with the necessary code

### annotation_path
This is the path to the annotation file

### model_path
If you have a model file available of the form `{model_name}.pt`
The default values is None

# Training

This is an example of how to train from scratch:

```bash
python code/src/entry.py --mode=train --root_dir=nycu-hw2-data --annotation_path=nycu-hw2-data
```

# Inference

This is an example of how run inference

```bash
python code/src/entry.py --mode=infer --root_dir=nycu-hw2-data --annotation_path=nycu-hw2-data --model_path=model.pt
```

# Performance Snapshot
[![Result](leaderboard.png)](result.png)

