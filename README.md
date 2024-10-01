# WavMCVM
Wavelet Multi-Contrast Vision Mamba SR Network
# Abstract
As an abnormal proliferation of cells within the central nervous system, brain tumors pose a significant threat to both neurological function and overall health. Magnetic Resonance Imaging (MRI) plays a crucial role in the detection and monitoring of brain tumors by providing high-resolution images of various brain tissues. However, acquiring high-resolution MRI images, particularly T2-weighted images, is often time-consuming, which can result in image quality degradation due to motion artifacts or signal decay. To address this issue, we propose a novel Wavelet Multi-Contrast Vision Mamba Super-resolution Network, termed WavMCVM. Our approach offers two main advantages: (1) We introduce a Feature Fusion Unit (FFU) that leverages residual connections, deep convolutional layers, and an adaptive weighting mechanism, enabling more effective fusion of reference image information while maintaining a low parameter count. (2) Unlike previous methods based on CNNs and Transformers, we are the first to integrate the Mamba model into the multi-contrast MRI super-resolution field. By incorporating wavelet transform, we utilize high-frequency information to guide the Mamba blocks in achieving more precise texture reconstruction, significantly reducing both parameter count and inference time. We evaluate our method on publicly available datasets, including BraTs2020 and IXI, and the experimental results demonstrate that our framework significantly outperforms existing single-contrast and multi-contrast super-resolution methods in the multi-contrast MRI super-resolution task.
# Main Environments
pip install -r requirements.txt
# Prepare the dataset
BraTs2020 as an example:
- **datasets/**
  - **BraTs2020/**
    - **train/**
      - **T2/**: Contains low-resolution T2-weighted MRI images
      - **T1/**: Contains corresponding high-resolution T1-weighted MRI images
    - **val/**
      - **T2/**: Contains validation set of T2-weighted MRI images
      - **T1/**: Contains validation set of T1-weighted MRI images
# Train the WavMCVM
python train.py
