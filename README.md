# ASTAR_internship

This repository contains the code for the preprocessing pipeline using **Fourier Domain Adaptation (FDA)** and **Optimal Transport (OT)**.  
The goal is to improve the generalization of AI models across different scanners by aligning either the visual appearance of images or their internal representations (embeddings).  

- The **Fourier_Domain_Adaptation** folder provides code to migrate the low-frequency components of images from one scanner to another.  
- The **optimal_transport** folder contains the code for training neural networks with OT-based loss functions, as well as visualizations of outputs from different AI models.  
- The pipeline also includes code for generating embeddings and evaluating performance with linear probing.

---

## Results

| Method                          | Metric       | Baseline | Improved   |
| ------------------------------- | ------------ | -------- | ---------- |
| Fourier Domain Adaptation (FDA) | **F1-score** | 0.8385   | **0.8427** |
| Optimal Transport (unsupervised) | **Accuracy** | 68.49%   | **75.26%** |
| Selective OT (supervised) + no detach       | **Accuracy** | 68.49%        | **73.77%** |
| Multi-scanner OT (naïve) + FDA       | **Accuracy** | 75.45%   | **78.90%** |


--- 

## 📖 Background

### Fourier Domain Adaptation and Optimal Transport for AI Generalization  
**Image and Embedding Migration**  
*Léo Leroy, Agency for Science, Technology and Research (A*STAR)*  
*September 2025*

**Abstract**  
The generalization of AI models for prostate cancer detection across different scanners remains a significant challenge. Models trained on a single scanner often fail to generalize due to scanner-specific artifacts, such as variations in color intensity, which are irrelevant to biological features.  

This work addresses this issue through **static domain adaptation**, aiming to align the appearance or internal representations (embeddings) of images from multiple scanners to a target reference scanner.  

Two approaches are explored:  

1. **Fourier Domain Adaptation (FDA):**  
   - A preprocessing pipeline that swaps low-frequency components of images to migrate their visual appearance to a target scanner.  
   - This yields a modest improvement in the F1-score (from **0.8385 → 0.8427**).  

2. **Optimal Transport (OT):**  
   - An OT-based loss function that aligns embeddings of image patches from different scanners to one reference scanner.  
   - This significantly enhances performance on unseen scanners, improving accuracy from **68.49% → 75.26%** without substantially increasing training time.  
   - A supervised variant, **Selective OT**, reaches **72.25% global accuracy** and  **73.77% global accuracy** without 'detach'. 
   - Extending OT to multiple scanners remains challenging, but even a naïve implementation improved accuracy from **75.45% → 78.9%**.  

---

## 🚀 Pipeline Guideline

1. Choose the correct FDA function.  
2. Generate the DeepLake dataset with the chosen transformation.  
3. Generate embeddings.  
4. Compute performance with linear probing.  

---

## 🔧 Installation

Clone this repository:
```bash
git clone https://github.com/Leooryx/ASTAR_internship.git
cd ASTAR_internship

