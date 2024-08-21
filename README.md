<div align="left">
  <a href="https://arxiv.org/abs/2401.14283">
    <img src="https://github.com/LeakDetectAI/automl-qild/blob/main/docs/source/logo.png" alt="Paper" width="300" height="300">
  </a>
</div>


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/LeakDetectAI/AutoMLQuantILDetect/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/automlquantildetect/badge/?version=latest)](https://automlquantildetect.readthedocs.io/?badge=latest)
[![Paper](https://img.shields.io/badge/arXiv-2401.14283-red)](https://arxiv.org/abs/2401.14283)

### AutoML Approaches to Quantify and Detect Leakage
The <strong>AutoMLQuantILDetect</strong> package utilizes AutoML approaches to accurately detect and quantify system information leakage.
We also provide different approaches to estimate mutual information (MI) within systems that release classification datasets to quantify system information leakage.
By leveraging state-of-the-art statistical tests, it precisely quantifies mutual information (MI) and effectively detects 
information leakage within classification datasets. With <strong>AutoMLQuantILDetect</strong>, users can confidently and 
comprehensively address the critical challenges of quantification and detection in information leakage analysis.


### 🛠️ Installation

The latest release version of AutoMLQuantILDetect can be installed from GitHub using the following command:

```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install git+https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
```

Alternatively, you can clone the repository and install AutoMLQuantILDetect using:

```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
git clone https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
cd AutoMLQuantILDetect
conda create --name ILD python=3.10
conda activate ILD
```
```
python setup.py install
```
**OR**
```
pip install -r requirements.txt
pip install -e ./
```

## ⚙️ Quickstart Guide
You can use `AutoMLQuantILDetect` in different ways.
Quite a few classifiers and AutoML tools already exist that can be used to estimate mutual information using the log-loss and the accuracy of the learned model.


### 📈 Fit a Classifier to Estimate MI
Fit a ClassficationMIEstimator on a synthetic dataset using a random forest, estimate mutual information using the log-loss and the accuracy of the learned model and compare it with the ground-truth mutual information.
You can find similar example code snippets in
**examples/**.

```python
from sklearn.metrics import accuracy_score
from autoqild.dataset_readers.synthetic_data_generator import SyntheticDatasetGenerator
from autoqild.mi_estimators.mi_estimator_classification import ClassificationMIEstimator
from autoqild.utilities._constants import LOG_LOSS_MI_ESTIMATION, MID_POINT_MI_ESTIMATION

# Step 1: Generate a Synthetic Dataset
random_state = 42
n_classes = 3
n_features = 5
samples_per_class = 200
flip_y = 0.10  # Small amount of noise

dataset_generator = SyntheticDatasetGenerator(
    n_classes=n_classes,
    n_features=n_features,
    samples_per_class=samples_per_class,
    flip_y=flip_y,
    random_state=random_state
)

X, y = dataset_generator.generate_dataset()

print(f"Generated dataset X shape: {X.shape}, y shape: {y.shape}")

# Step 2: Estimate Mutual Information using ClassficationMIEstimator
mi_estimator = ClassificationMIEstimator(n_classes=n_classes, n_features=n_features, random_state=random_state)

# Fit the estimator on the synthetic dataset
mi_estimator.fit(X, y)

# Estimate MI using log-loss
estimated_mi_log_loss = mi_estimator.estimate_mi(X, y, method=LOG_LOSS_MI_ESTIMATION)
estimated_mi_mid_point = mi_estimator.estimate_mi(X, y, method=MID_POINT_MI_ESTIMATION)
# Step 3: Calculate Accuracy of the Model
y_pred = mi_estimator.predict(X)
accuracy = accuracy_score(y, y_pred)

# Step 4: Compare with Ground-Truth MI
ground_truth_mi = dataset_generator.calculate_mi()

# Summary of Results
print("##############################################################")
print(f"Ground-Truth MI: {ground_truth_mi}")
print(f"Estimated MI (Log-Loss): {estimated_mi_log_loss}")
print(f"Estimated MI (Mid-Point): {estimated_mi_mid_point}")
print(f"Model Accuracy: {accuracy}")

>> Generated
dataset
X
shape: (600, 5), y
shape: (600,)
>>  ##############################################################
>> Ground - Truth
MI: 1.1751928845077875
>> Estimated
MI(Log - Loss): 1.3193094645863748
>> Estimated
MI(Mid - Point): 1.584961043823006
>> Model
Accuracy: 1.0

```

 
### 💬 Cite Us
If you use this toolkit in your research, please cite our paper available on arXiv:
```
@article{gupta2024information,
  title={Information Leakage Detection through Approximate Bayes-optimal Prediction},
  author={Pritha Gupta, Marcel Wever, and Eyke Hüllermeier},
  year={2024},
  eprint={2401.14283},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## 📧 Contact Information
For any questions or feedback, please contact Pritha Gupta at prithagupta15@gmail.com.


