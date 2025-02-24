#!/bin/bash

BASE_DIR="project"

mkdir -p $BASE_DIR/data
mkdir -p $BASE_DIR/models
mkdir -p $BASE_DIR/utils
mkdir -p $BASE_DIR/experiments
mkdir -p $BASE_DIR/notebooks

touch $BASE_DIR/data/__init__.py
touch $BASE_DIR/models/__init__.py
touch $BASE_DIR/utils/__init__.py

touch $BASE_DIR/data/load_data.py
touch $BASE_DIR/models/baseline_cnn.py
touch $BASE_DIR/models/mc_dropout.py
touch $BASE_DIR/models/deep_ensemble.py
touch $BASE_DIR/models/evidential.py
touch $BASE_DIR/utils/metrics.py
touch $BASE_DIR/utils/visualization.py

touch $BASE_DIR/experiments/train_baseline.ipynb
touch $BASE_DIR/experiments/train_mc_dropout.ipynb
touch $BASE_DIR/experiments/train_deep_ensemble.ipynb
touch $BASE_DIR/experiments/train_evidential.ipynb
touch $BASE_DIR/notebooks/Data_Exploration.ipynb
touch $BASE_DIR/notebooks/Uncertainty_Analysis.ipynb

touch $BASE_DIR/requirements.txt
touch $BASE_DIR/README.md
touch $BASE_DIR/.gitignore

echo "Project structure created successfully in '$BASE_DIR'."
