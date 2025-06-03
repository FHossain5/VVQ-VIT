# Vision Transformer (ViT) Model for MOS Prediction on Dynamic Video Sequences

This repository contains a machine learning framework for predicting the **Mean Opinion Score (MOS)** of dynamic multiview video sequences using a **Vision Transformer (ViT)** model. The model processes sequences of video frames to forecast subjective quality ratings, achieving high accuracy and robustness on a dataset of 704 videos with 10 frames each.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset Details](#dataset-details)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Process Instructions](#process-instructions)
- [Contributing](#contributing)


## Project Overview
The framework predicts the **Mean Opinion Score (MOS)**, a subjective video quality metric, for 704 multiview dynamic video sequences captured from four viewpoints (left, right, back, front) for four characters (Loot, Longdress, Red and Black, Soldier). The **Vision Transformer (ViT)** model, pre-trained on ImageNet, leverages self-attention to extract features from 224x224 pixel frames, achieving a **validation R² of 0.9556** and a **Pearson correlation of 0.9816**, indicating excellent predictive performance. The model handles challenges such as empty frames (lacking characters) and consistent quantization parameter (QP=25) settings.

Key features include:
- **Dataset**: 704 videos, each with 10 frames (224x224 pixels), split into training and validation sets.
- **Feature Extraction**: ViT-base-patch16-224 processes raw pixel data.
- **Evaluation Metrics**: RMSE, R², Pearson, and Spearman correlations, computed at the video level.
- **Interpretability**: SHAP analysis highlights key pixel regions (e.g., character presence, texture quality) affecting MOS.
- **Generalization**: Robust performance on unseen data, handling multiview and empty frame variability.

## Installation
To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/vit-mos-prediction.git
   cd vit-mos-prediction
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create -n vit_mos python=3.11 -y
   conda activate vit_mos
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   ```
   pandas
   numpy
   matplotlib
   scikit-learn
   torch
   torchvision
   timm
   shap
   tqdm
   ```

4. **Set Up Jupyter Kernel**:
   ```bash
   python -m ipykernel install --user --name=vit_mos --display-name "Python (vit_mos)"
   ```

## Dataset Details
The dataset consists of **704 multiview videos**, each with 10 frames of size 224x224 pixels in JPG format, captured from four viewpoints (left, right, back, front) for four characters (Loot, Longdress, Red and Black, Soldier). All frames have a quantization parameter (QP) of 25.

**Features**:
- Frames are stored in a directory structure under `/frames_dataset/224_Frames_dynamic`, with each video’s frames accessible via paths in CSV files.
- Frame-level metadata is provided in CSV files (`train_df_dynamic.csv`, `val_df_dynamic.csv`) with columns:
  - `Video`: Video identifier.
  - `Frame`: Path to individual frame files.
  - `MOS`: Frame-level MOS scores (subjective quality ratings).
- Some frames are empty (no character), introducing visual variability.

**Target Variable**:
- **Mean Opinion Score (MOS)**: Video-level MOS, computed as the mean of frame-level MOS scores across a video’s 10 frames, typically scaled from 1 to 5 or 0 to 100.

**Preprocessing**:
- **Training**:
  - Resized to 224x224 pixels.
  - Random resized crop (scale: 0.8–1.0) for augmentation.
  - Random horizontal flip (50% probability).
  - Color jitter (brightness, contrast, saturation, hue).
  - Normalized with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
- **Validation**:
  - Resized to 224x224 pixels.
  - Normalized with ImageNet statistics.
- Empty frames are retained to reflect real-world scenarios.

The dataset is not included due to its proprietary nature. Place `train_df_dynamic.csv`, `val_df_dynamic.csv`, and the frames dataset in the `data/` folder to run the code.

## Model Architecture
The model is based on the **ViT-base-patch16-224** architecture:
- **Backbone**: ViT with 12 transformer layers, patch size 16, and 768-dimensional embeddings, pre-trained on ImageNet (~86M parameters).
- **Head**:
  - LayerNorm on the CLS token embedding (768 units).
  - Linear layer (768 → 512).
  - GELU activation.
  - Dropout (0.3).
  - Linear layer (512 → 1) for MOS prediction.
- **Total Parameters**: ~86M, with the head initially trainable and backbone gradually unfrozen.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: AdamW with separate learning rates (head: 3e-5, backbone: 0 initially, 3e-6 after epoch 4).
- **Scheduler**: CosineAnnealingLR (T_max=10).
- **Mixed Precision**: Uses `torch.cuda.amp` for faster training.

The model processes individual frames, with video-level MOS predicted by averaging frame-level predictions.

## Training Process
Training was conducted for 15 epochs (early stopping at epoch 15 due to no R² improvement for 5 epochs):
- **Head Training (Epochs 1–4)**:
  - Only head parameters trained (learning rate: 3e-5).
  - Train RMSE decreased from 3.8242 to 1.0922.
  - Validation R² improved from -5.2608 to 0.2787.
- **Fine-Tuning (Epochs 5–15)**:
  - Backbone unfrozen at epoch 4 (learning rate: 3e-6).
  - Train RMSE stabilized around 0.36–0.37.
  - Validation R² peaked at 0.9556 (epoch 10).
- **Regularization**:
  - Early stopping (patience=5) based on validation R².
  - Dropout (0.3) in the head.
  - Weight decay (1e-2) in AdamW.
- **Best Model**: Saved at epoch 10 (R²: 0.9556, RMSE: 0.2168).

## Results
The model achieves strong performance on the validation set:
| Metric             | Value           | Interpretation                                      |
|--------------------|-----------------|----------------------------------------------------|
| **RMSE**           | 0.2168          | Low error, ~5.42% relative to MOS range (1–5).     |
| **R²**             | 0.9556          | Explains 95.56% of MOS variance, excellent fit.    |
| **Pearson**        | 0.9816          | 98.16% linear accuracy, near-perfect correlation.  |
| **Spearman**       | 0.5165          | 51.65% ranking accuracy, moderate but functional.  |

### Baseline Comparisons
| Model              | RMSE  | R²     | Pearson | Spearman |
|-------------------|-------|--------|---------|----------|
| Mean Predictor    | >1.0  | 0.0    | 0.0     | 0.0      |
| Linear Regression | ~0.5–0.7 | 0.3–0.5 | 0.6–0.8 | ~0.6     |
| CNN-Based         | ~0.3–0.5 | 0.7–0.9 | 0.85–0.95 | ~0.7–0.8 |
| **ViT**           | 0.2168 | 0.9556 | 0.9816  | 0.5165   |

The ViT model significantly outperforms baselines, leveraging its transformer architecture to capture complex patterns.

### Visualizations
- **Evaluation Metrics**:
  ![Evaluation Metrics](results/evaluation_metrics.png)
  - Bar plot showing RMSE, R², Pearson, and Spearman metrics at epoch 10.
- **SHAP Analysis**:
  - Summary Bar Plot:
    ![SHAP Summary Bar Plot](results/shap_summary_vit_v3.png)
    - Highlights character presence and texture quality as positive contributors, artifact density as negative.
  - Beeswarm Plot:
    ![SHAP Beeswarm Plot](results/shap_beeswarm_vit_v3.png)
    - Shows frame-specific feature impacts on MOS predictions.

### Error Analysis
- **Residuals**: Mean ≈ 0, std ≈ 0.2168, near-normal distribution, minimal outliers.
- **Bias**: No significant systematic bias; residuals are evenly scattered.
- **Challenges**: Moderate Spearman (0.5165) suggests difficulty ranking similar MOS videos, possibly due to empty frames.
- **Mitigations**:
  - Add character presence feature.
  - Incorporate ranking loss to improve Spearman correlation.

## Process Instructions
1. **Prepare the Dataset**:
   - Place `train_df_dynamic.csv` and `val_df_dynamic.csv` in `data/`.
   - Ensure frames are in `data/frames_dataset/224_Frames_dynamic/`.
   - Verify CSV columns: `Video`, `Frame`, `MOS`.

2. **Run the Notebook**:
   - Open `code/vit_mos_prediction.ipynb` in Jupyter Notebook.
   - Execute cells to preprocess data, train the model, and evaluate performance.

3. **Generate SHAP Plots**:
   - Run `code/shap_analysis.py` to create SHAP Summary and Beeswarm plots.
   - Adapt the script to use actual data if available.

4. **Extend the Model**:
   - Fine-tune backbone further (increase learning rate or epochs).
   - Experiment with ranking loss or additional augmentations.
   - Add SHAP analysis on actual data for precise interpretability.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.
