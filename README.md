<a href="https://github.com/LeakDetectAI/AutoMLQuantILDetect/blob/master/LICENSE">
  <img src="https://github.com/LeakDetectAI/automl-qild/blob/main/images/apache.png" alt="License" width="100" height="60">
</a>
<a href="https://arxiv.org/abs/2401.14283">
  <img src="https://github.com/LeakDetectAI/automl-qild/blob/main/images/logo.png" alt="Paper" width="100" height="60">
</a>


### AutoML Approaches to Quantify and Detect Leakage

The AutoMLQuantILDetect package utilizes AutoML approaches to detect and quantify system information leakage. It is an advanced toolkit that leverages the power of Automated Machine Learning (AutoML) to quantify information leakage accurately. This package estimates mutual information (MI) within systems that release classification datasets. By leveraging state-of-the-art statistical tests, it precisely quantifies mutual information (MI) and effectively detects information leakage within classification datasets. With AutoMLQuantILDetect, users can confidently and comprehensively address the critical challenges of quantification and detection in the realm of information leakage analysis.

### Installation

The latest release version of AutoMLQuantILDetect can be installed from GitHub using the following command:

```
pip install git+https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
```

Alternatively, you can clone the repository and install AutoMLQuantILDetect using:

```
python setup.py install
```

### Dependencies

AutoMLQuantILDetect depends on the following libraries:
- AutoGLuon
- TabPFN
- Pytorch
- Tensorflow
- NumPy
- SciPy
- matplotlib
- Scikit-learn
- tqdm
- pandas (required for data processing and generation)

### Citing automl-qild

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


