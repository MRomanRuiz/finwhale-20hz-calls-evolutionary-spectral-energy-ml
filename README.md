# Fin Whale 20 Hz Calls - Evolutionary Spectral Energy ML

## Project Overview

This repository contains a comprehensive MATLAB-based system for acoustic event detection and classification, specifically designed for fin whale 20 Hz calls. The project combines evolutionary algorithms with machine learning techniques to automatically detect and classify acoustic events.

### Key Features
- Automatic detection threshold calculation using evolutionary algorithms (1+1)-ES
- Spectral feature extraction from acoustic signals
- k-NN and SVM classifier training and evaluation
- Evolutionary threshold optimization

## Requirements

- **MATLAB** with the following toolboxes:
  - Signal Processing Toolbox (`spectrogram`, `audioread`)
  - Statistics and Machine Learning Toolbox (`fitcknn`, `fitcsvm`, `crossval`, `clusterdata`)
  - Audio I/O functions (`audioDatastore`)
- Audio files in `.aif` format with naming convention `00000XXX.aif`

## Usage

1. **Configure paths** in the scripts to point to your audio data directory
2. **Training phase**: Run `TRAINING_KNN.m` or `TRAINING_SVM.m` to train classifiers
3. **Classification phase**: Run `CLASSIFICATION_KNN.m` or `CLASSIFICATION_SVM.m` to classify new data
4. **Post-processing**: Run `post_processing.m` to refine results
5. **Evolutionary tuning**: Use scripts in the `Evolutionary algorithm/` folder to optimize thresholds

## Repository Structure

### Root Level Files

- **CLASSIFICATION_KNN.m** - k-Nearest Neighbors classification script
- **CLASSIFICATION_SVM.m** - Support Vector Machine classification script
- **TRAINING_KNN.m** - k-NN model training script
- **TRAINING_SVM.m** - SVM model training script
- **post_processing.m** - Post-processing and refinement of classification results
- **dataGT.mat** - Ground truth data (MATLAB binary format)
- **KNNModel_0801.mat** - Pre-trained k-NN model
- **SVMModel_1301.mat** - Pre-trained SVM model
- **resultsKNN_1401.mat** - k-NN classification results
- **resultsSVM_1401.mat** - SVM classification results
- **resultsCV.mat** - Additional classification results obtained via Visual Computer techniques (see [CV detection algorithm](https://github.com/MRomanRuiz/finwhale-20hz-calls-cv-detection)) for more information

### Evolutionary Algorithm Directory

The `Evolutionary algorithm/` folder contains specialized scripts for threshold optimization and parameter tuning:

- **extract_features.m** - Extract spectral features from audio signals
- **threshold_testing.m** - Test and evaluate detection thresholds
- **evaluate_KNNthreshold.m** - Evaluate k-NN threshold performance
- **evaluate_SVMthreshold.m** - Evaluate SVM threshold performance
- **KNN_tuning.m** - Hyperparameter tuning for k-NN
- **SVM_tuning.m** - Hyperparameter tuning for SVM
- **trainKNN.m** - Train k-NN model with specified parameters
- **trainSVM.m** - Train SVM model with specified parameters
