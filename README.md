# Intrusion Detection using Machine Learning

An Intrusion Detection System (IDS) monitors network traffic to detect unauthorized or malicious activities. Instead of relying on static rule-based systems, this project applies supervised learning techniques to automatically learn the behavior of normal and attack traffic.

This project implements a Machine Learning-based Intrusion Detection System (IDS) using the NSL-KDD dataset. The system identifies and classifies network intrusions using classical and deep learning algorithms. The objective is to analyze network traffic, detect abnormal behavior, and compare model performances across binary and multi-class classification tasks.

The implementation uses a combination of classical machine learning algorithms (SVM, KNN, LDA) and deep learning architectures (MLP, LSTM).
Both **binary** (Normal vs Attack) and **multi-class** (different attack categories) tasks are addressed.


---


## Dataset

**Dataset Used:** NSL-KDD
**Source:** [Canadian Institute for Cybersecurity – NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)

The dataset contains 41 network traffic features (e.g., protocol type, service, bytes transferred, and flag). Each record is labeled as either *normal* or one of multiple attack types (DOS, Probe, R2L, U2R).


---


## Workflow Summary

### 1. Data Preprocessing

* Load the raw dataset (`KDDTrain+.txt`)
* Assign column names to match NSL-KDD features
* Encode categorical variables (`protocol_type`, `service`, `flag`)
* Normalize numerical attributes using `StandardScaler`
* Split into training and testing sets
* Create binary and multi-class label datasets

### 2. Model Training

| Model                              | Type          | Library      | Description                                      |
| ---------------------------------- | ------------- | ------------ | ------------------------------------------------ |
| K-Nearest Neighbors (KNN)          | Classical ML  | scikit-learn | Classifies based on nearest neighbors            |
| Linear Discriminant Analysis (LDA) | Classical ML  | scikit-learn | Projects data to maximize class separation       |
| Support Vector Machine (SVM)       | Classical ML  | scikit-learn | Constructs an optimal separating hyperplane      |
| Multi-Layer Perceptron (MLP)       | Deep Learning | Keras        | Feed-forward neural network with dense layers    |
| Long Short-Term Memory (LSTM)      | Deep Learning | Keras        | Sequential model capturing temporal dependencies |

### 3. Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Visualization outputs include:

* Accuracy/Loss curves (MLP, LSTM)
* ROC curves
* Confusion matrices
* Prediction distribution pie charts

Plots are available in the `plot_figures/` directory.


---


## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/deepanshi211/INTRUSION-DETECTION-APR-PROJECT.git
   cd INTRUSION-DETECTION-APR-PROJECT
   ```

2. Install dependencies:

   ```bash
   pip install keras scikit-learn pandas numpy matplotlib
   ```

3. Run the notebook:

   ```bash
   jupyter notebook Notebooks/intrusion_detection.ipynb
   ```

The notebook will preprocess the dataset, train models, evaluate results, and save outputs automatically.


---



## Collaborators

* 2201CS90 – Medha Aggarwal
* 2201AI47 – Ankit Singh
* 2201AI15 – Harshit Tomar
* 2201AI16 – Himani Yadav
* 2201AI45 – Yash Kamdar
* 2201AI56 – Sanskruti Kulkarni
* 2201CS18 – Anthadupula Akshaya Tanvi
* 2201CS85 – Deepanshi Verma
* 2201CS61 – P. Sai Lasya
* 2201CS32 – Isha Jaiswal
* 2201CS81 – Ravina
* 2201CS82 – Sanjana Mooli

This is a **closed academic project**.
No external collaboration, pull requests, or code contributions are required.
The repository is maintained for documentation and reference purposes only.

---
