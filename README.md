# DualFormer for Blood Glucose Prediction

This repository provides a PyTorch implementation of **DualFormer**, a Transformer-based model for blood glucose time-series prediction.

## Overview

DualFormer predicts future glucose values from historical continuous glucose monitoring data. The model uses a dual-branch structure to capture both short-term glucose fluctuations and long-term temporal trends.

## Project Structure

```text
.
├── main.py
├── models/
│   └── DualFormer.py
├── layers/
│   ├── Embed.py
│   ├── SelfAttention_Family.py
│   └── utils.py
└── utils/
    ├── _init_.py
    ├── data.py
    ├── masking.py
    ├── metet.py
    ├── resample.py    
    └── timefeatures.py
```

## Key Features

* Dual-scale patch embedding
* Short-term and long-term Transformer experts
* Gating-based feature fusion
* Time feature and type embedding
* Patient-balanced training sampler
* Clinically weighted focal loss

## Training

Run the training script with default settings:

```bash
python main.py
```

Example with custom parameters:

```bash
python main.py --data_path ./dataset --seq_len 96 --pred_len 24 --batch_size 128
```

