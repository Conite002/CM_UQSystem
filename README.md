# 🧠 Uncertainty Quantification on MNIST

This project implements and compares **uncertainty quantification methods** in deep neural networks using the **MNIST dataset**. The project includes:

✅ **Baseline CNN** for classification.  
✅ **Bayesian Neural Networks (MC Dropout)**.  
✅ **Deep Ensembles**.  
✅ **Evidential Deep Learning** (predicting Dirichlet parameters).  

---

## 📁 **Project Structure**

project/ 
├── data/ 
# Data processing scripts 
│
├── init.py 
│ 
└── load_data.py 
├── models/ 
# Model architectures │ 
├── init.py │ 
├── baseline_cnn.py │ 
├── mc_dropout.py │ 
├── deep_ensemble.py │ 
└── evidential.py 
├── utils/ # Utility functions (metrics, plots) │ 
├── init.py │ ├── metrics.py │ 
└── visualization.py 
├── experiments/ 
# Training scripts │ 
├── train_baseline.ipynb │ 
├── train_mc_dropout.ipynb │ 
├── train_deep_ensemble.ipynb │ 
└── train_evidential.ipynb 
├── notebooks/ 
# Analysis and reports │ 
├── Data_Exploration.ipynb │ 
└── Uncertainty_Analysis.ipynb 
├── requirements.txt # Dependencies 
├── README.md # Project documentation 
└── .gitignore # Files to ignore

# 🏗 Project Workflow

🔹 **Data Preparation**
Load MNIST dataset using data/load_data.py
Normalize images, reshape for CNN input, and split into training/testing sets.

🔹 **Model Training**
Train the Baseline CNN, MC Dropout, Deep Ensembles, and Evidential Network using the scripts in experiments/.

🔹 **Evaluation & Uncertainty Analysis**
Compute accuracy, predictive entropy, mutual information, and calibration metrics (ECE).
Compare uncertainty quantification across models.

🔹 **Experiment Tracking**
(Optional) Use MLflow or Weights & Biases to track hyperparameters and results.
📊 Model Evaluation and Calibration
✅ Implemented Metrics:

**Accuracy**: Standard performance metric.

**Predictive Entropy**: Measures confidence in predictions.

**Mutual Information (MI)**: Estimates model uncertainty.

**Expected Calibration Error (ECE)**: Measures calibration quality.
AUROC for OOD Detection: Detects out-of-distribution samples.
# 📌 Visualization:
Reliability Diagrams to visualize model calibration.
Uncertainty heatmaps over MNIST test images.
Comparison of uncertainty distributions (box plots, histograms).
