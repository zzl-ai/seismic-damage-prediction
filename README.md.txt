# Subjective Logic-Driven Building Seismic Damage Prediction with Uncertainty Quantification

This repository provides the complete implementation for the paper:
**Subjective Logic-Driven Building Seismic Damage Prediction and Uncertainty Analysis**

## Overview
This project proposes a novel evidential fusion framework based on subjective logic and Dempster-Shafer (D-S) evidence theory to predict building damage grades after earthquakes and quantify prediction uncertainty.

The framework integrates three complementary base learners:
- XGBoost
- CatBoost
- TabTransformer

It outputs not only damage grade predictions but also interpretable belief and uncertainty values for each decision.

## File Structure
- main.py                    # Main code for model training, evidence fusion, and evaluation
- intensity_normal.csv        # Preprocessed dataset (feature selection + normalization)
- requirements.txt            # Python environment dependencies
- README.md                   # Documentation

## Dataset
The dataset is derived from the 2015 Nepal earthquake building damage survey.
The provided `intensity_normal.csv` includes:
- Feature selection (top important features retained)
- Min-Max normalization
- Cleaned and filtered samples
- Train/validation/test split ratio: 7:1:2

## Reproducibility
All hyperparameters, random seeds, model settings, and fusion formulas are **fully consistent with the paper**.
Slight numerical variations (within ±1%) may arise from different hardware or software environments, but the core conclusions remain stable and reproducible.

## How to Run
1. Install required packages:
   pip install -r requirements.txt

2. Run the main script:
   python main.py

## Pipeline
1. Data loading and stratified train/val/test split (7:1:2)
2. Base model training and raw evidence extraction
3. Belief-uncertainty (B-U) pair generation using Dirichlet prior
4. Weight correction using validation accuracy
5. Multi-view D-S evidence fusion
6. Uncertainty-aware hierarchical classification (5-class / 3-class)
7. Performance evaluation and result output

## Citation
If you use this code in your research, please cite our paper.

## Contact
For questions or issues, please contact the authors.