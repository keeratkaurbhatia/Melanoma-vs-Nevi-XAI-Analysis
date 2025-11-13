# Bridging the Gap Between AI and Clinical Reasoning: An Explainable AI Approach to Skin Cancer Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive study evaluating multiple Explainable AI (XAI) techniques for melanoma classification using EfficientNetB3, with quantitative analysis of clinical relevance and method consistency.

## 🔬 Overview

This project addresses the critical need for transparency and trust in AI-assisted melanoma diagnosis. While deep learning models achieve impressive accuracy in skin lesion classification, their "black box" nature limits clinical adoption. Our work bridges this gap by:

- Implementing and comparing **8 state-of-the-art XAI techniques**
- Developing **automated quantitative metrics** for clinical relevance
- Providing **objective evaluation** of explanation quality
- Analyzing **inter-method consistency** across different XAI approaches

### Why This Matters

- **Clinical Trust**: Dermatologists need to understand *why* a model makes a diagnosis
- **Patient Safety**: False negatives in melanoma detection can be fatal
- **Regulatory Compliance**: Healthcare AI requires interpretability for deployment
- **Bias Detection**: XAI reveals model focus areas and potential biases

## ✨ Key Features

### 🎯 Model Architecture
- **EfficientNetB3** with transfer learning
- Two-stage training with cosine annealing
- Custom preprocessing pipeline (Contrast Stretching, Gamma Correction, CLAHE)
- Optimized threshold (0.400) for balanced precision-recall

### 🔍 XAI Techniques
- **Gradient-based**: Vanilla Gradients, SmoothGrad, Integrated Gradients, SHAP
- **CAM-based**: Grad-CAM, Grad-CAM++
- **Perturbation-based**: LIME, Occlusion Sensitivity

### 📊 Novel Evaluation Framework
- **Clinical Relevance Metrics**: Coverage, Precision, Attention Ratio
- **Consistency Analysis**: Pairwise correlation, SSIM, IoU
- **Automated Lesion Masking**: LAB color space + Otsu thresholding

## 🏗️ Architecture

```
Input Image (300×300)
    ↓
Preprocessing (Normalization)
    ↓
EfficientNetB3 (pretrained)
    ↓
Custom Head (GAP + Dropout + Dense)
    ↓
Binary Classification (Melanoma vs Nevi)
    ↓
XAI Attribution Maps
    ↓
Quantitative Evaluation
```

### Model Performance
- **Accuracy**: 77%
- **Precision**: 71%
- **Sensitivity/Recall**: 91% ⭐
- **Specificity**: 63%
- **F1-Score**: 80%

> High sensitivity prioritizes patient safety by minimizing false negatives.

## 📁 Dataset

**HAM10000** (Human Against Machine with 10000 training images)
- 10,015 dermatoscopic images
- 7 diagnostic classes
- >50% histopathology-confirmed
- Focus: Binary classification (Melanoma vs Nevi)

**Source**: [HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)


## 🔍 XAI Methods Implemented

### Gradient-Based Methods

#### 1. Vanilla Gradients (Saliency Maps)
```python
gradients = compute_gradients(model, image, target_class)
saliency_map = np.abs(gradients).max(axis=-1)
```

#### 2. SmoothGrad
- Averages gradients over 50 noisy samples
- Reduces noise, improves stability

#### 3. Integrated Gradients
- Axiom-satisfying attribution
- 100-step path integration
- Multiple baselines (black, white, gray)

#### 4. SHAP (DeepSHAP)
- Shapley value approximation
- 50 interpolation steps

### Class Activation Mapping

#### 5. Grad-CAM
- Targets: `efficientnetb3/top_conv`
- Global average pooling of gradients

#### 6. Grad-CAM++
- Enhanced localization with weighted partial derivatives

### Perturbation-Based Methods

#### 7. LIME
- 100 superpixels (SLIC algorithm)
- 1000 perturbations
- Ridge regression surrogate model

#### 8. Occlusion Sensitivity
- Patch size: 60×60 pixels
- Stride: 30 pixels
- Direct output change measurement

### Key Findings

1. **Grad-CAM++ Superiority**: Highest coverage and precision
2. **LIME-SmoothGrad Correlation**: r=0.966 despite different theoretical bases
3. **Method Fragmentation**: Low correlation between most methods
4. **Gradient Method Instability**: Sensitive to hardware/precision


## 🔬 Research Highlights

### Novel Contributions

1. **Automated Lesion Masking**: LAB color space + Otsu thresholding for objective evaluation
2. **Multi-Metric Framework**: Goes beyond visual inspection
3. **Comprehensive Comparison**: 8 XAI methods in unified pipeline
4. **Threshold Optimization**: F1-based optimal threshold (0.400)

### Clinical Implications

- **Actionable Insights**: Grad-CAM++ provides most clinically relevant explanations
- **Trust Building**: Quantitative metrics support clinical validation
- **Bias Detection**: Reveals model focus on non-diagnostic features
- **Decision Support**: High sensitivity reduces missed melanomas

## ⚠️ Limitations

1. **No Expert Validation**: Findings require dermatologist review
2. **Hardware Constraints**: TPU mixed-precision affects gradient stability
3. **Resolution**: 300×300 input limits fine-detail analysis
4. **Dataset Scope**: Binary classification only (Melanoma vs Nevi)

## 🚀 Future Work

- [ ] Clinical expert validation study
- [ ] High-resolution analysis on dedicated GPUs
- [ ] Multi-class extension (all 7 HAM10000 classes)
- [ ] Real-time deployment pipeline
- [ ] Standardized XAI benchmarks for medical imaging


## 👥 Authors

**Keerat Kaur** & **Hiya Trehan** (All authors contributed equally to this work.)

- Department of Artificial Intelligence and Data Sciences / Information Technology
- Indira Gandhi Delhi Technical University for Women, Delhi, India
- Contact: itskeeratkaur@gmail.com, hiyatrehan@gmail.com

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Tschandl et al. (2018)
- **EfficientNet**: Google Research (Tan & Le, 2019)
- **XAI Libraries**: TensorFlow, LIME, SHAP, scikit-image
- **Computing**: Google Colab (free TPU tier)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
