# PulsePitch: Advanced Speech Emotion Recognition
![Python](https://img.shields.io/badge/python-v3.10-blue.svg) ![version](https://img.shields.io/badge/version-10.0.0-green)

PulsePitch is a high-fidelity Speech Emotion Recognition (SER) framework that extracts emotional intent from vocal prosody using 1D/2D-CNN architectures. It features Z-Score normalization, aggressive silence trimming, and dual-input tensor synchronization to achieve high accuracy and sub-100ms inference latency.

## Description
This project was developed as part of my **Artificial Intelligence Internship** at **CodeAlpha**. It represents a high-fidelity implementation of Speech Emotion Recognition (SER), designed to demonstrate the practical application of deep learning in acoustic signal processing.

The framework utilizes a dual-model approach to acoustic analysis:
*   **1D-CNN Architecture**: Analyzes 40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) to identify short-term power spectrum patterns.
*   **2D-CNN Architecture (Mel-Spectrograms)**: Processes 2D visual representations of speech signals to extract salient emotional features via convolutional layers.

### Key Technical Innovations
*   **Signal Normalization**: Implementation of Z-Score scaling to ensure zero-mean and unit-variance, reducing classification bias.
*   **Acoustic Alignment**: Aggressive silence trimming with a 20dB threshold to ensure the models process high-density speech data.
*   **Multi-Tier Inference**: Simultaneously predicts across 3-tier (Sentiment), 6-tier (Standard), and 7-tier (Extended) emotion categories.
*   **Gender Identification**: Integrated neural logic for real-time gender classification based on vocal tract filter responses.

## Installation
It is recommended to use a virtual environment (Conda or venv) with Python 3.10 for optimal stability.
```sh
# Clone the repository
git clone [https://github.com/radhikapatil17/PulsePitch.git](https://github.com/radhikapatil17/PulsePitch.git)
cd PulsePitch

# Install dependencies
pip install -r requirements.txt
Usage
Run the dashboard locally using Streamlit:

Bash
streamlit run app.py

Datasets
The models were trained using benchmark emotional speech datasets:

Crema-D: Crowd-sourced Emotional Multimodal Actors Dataset.

Ravdess: Ryerson Audio-Visual Database of Emotional Speech and Song.

Savee: Surrey Audio-Visual Expressed Emotion.

Tess: Toronto emotional speech set.
