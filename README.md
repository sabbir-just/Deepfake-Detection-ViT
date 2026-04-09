# Deepfake Detection: A Comparative Analysis of CNN and Transformer-Based Methods

This repository contains the official implementation of our research project conducted at the **Department of Mathematics, Jashore University of Science and Technology (JUST)**.

## Abstract
Recent advances in generative modeling have enabled the synthesis of highly realistic facial images, commonly referred to as deepfakes. This project presents a controlled and reproducible comparative study of CNN architectures (ResNet-18, EfficientNet-B0) and a transformer-based model (ViT-B/16) within a unified, face-centric framework. 

## Key Findings
* [cite_start]**Superior Accuracy:** The ViT-B/16 model achieved a video-level accuracy of **98.20%** and an ROC-AUC of **0.9957** on the Celeb-DF v2 dataset[cite: 13, 187].
* [cite_start]**Global Attention:** Our research highlights the effectiveness of global self-attention in capturing spatially distributed manipulation artifacts compared to the local receptive fields of CNNs[cite: 31, 305].
* [cite_start]**Aggregation Stability:** We demonstrate that video-level mean aggregation significantly stabilizes frame-level predictions[cite: 11, 349].

## Repository Structure
* `training_vit_b16.py`: Core training script for the Vision Transformer using PyTorch.
* `preprocessing/`: Scripts for MTCNN face detection and frame sampling.
* `requirements.txt`: List of dependencies.

## Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Deepfake-Detection-ViT.git](https://github.com/YourUsername/Deepfake-Detection-ViT.git)# Deepfake-Detection-ViT
