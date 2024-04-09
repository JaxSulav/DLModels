# Facial Expression Classifier Collection

This repository hosts three innovative Streamlit applications designed for facial expression classification and image generation. Each app utilizes different deep learning models to perform tasks ranging from facial expression recognition to generative adversarial network-based image production.

## Apps Overview

1. **Face Expression Classifier** (`facecls_stream.py`)
   - Recognizes facial expressions from uploaded images.
2. **Face Expression Classifier with CNN** (`face_clscnn_stream.py`)
   - Advanced model using convolutional neural networks for more accurate expression recognition.
3. **WGANGP Streamlit App** (`wgangp_stream.py`)
   - Utilizes a Wasserstein GAN with gradient penalty to generate new facial images based on learned distributions.

## Installation

To run these apps locally, you need Python and several dependencies installed. Here are the steps to set up your environment:

### Prerequisites:

- Python 3.10
- pip install -r requirements.txt

### Prerequisites Environments Info:
There are two environments used in requirements.txt file. Use accordingly for tensorflow and pytorch models.

### Setup Local:
Clone the repository:

```bash
git clone https://github.com/JaxSulav/DLModels
cd DLModels
```

### Run Local:

```bash
streamlit run {desired_file}.py
```