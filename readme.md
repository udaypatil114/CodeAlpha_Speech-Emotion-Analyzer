# Speech Emotion Analyzer

![Python](https://img.shields.io/badge/python-v3.10-blue.svg)
![Machine Learning](https://img.shields.io/badge/MachineLearning-AI-green)

## Overview
Speech Emotion Analyzer is a machine learning based system that detects human emotions from speech audio using deep learning techniques. The project analyzes voice features such as pitch, tone, frequency, and energy to classify emotions in real time.

This project was developed as part of my Machine Learning Internship at CodeAlpha to demonstrate practical applications of artificial intelligence and audio signal processing.

---

## Features
- Real-time speech emotion detection
- Deep learning based CNN models
- MFCC and Mel-Spectrogram feature extraction
- Audio preprocessing and normalization
- Automatic silence trimming
- Multi-class emotion classification
- Interactive Streamlit dashboard
- Fast and efficient prediction system

---

## Technologies Used
- Python 3.10
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Streamlit
- Matplotlib

---

## Deep Learning Models

### 1D CNN Model
Processes MFCC audio features to identify emotional speech patterns.

### 2D CNN Model
Uses Mel-Spectrogram representations of audio for improved emotion classification.

---

## Emotion Categories
The system can recognize emotions such as:
- Happy
- Sad
- Angry
- Fear
- Neutral
- Surprise
- Disgust

---

## Installation

```bash
# Clone the repository
git clone https://github.com/udaypatil114/CodeAlpha_Speech-Emotion-Analyzer.git

# Open the project folder
cd CodeAlpha_Speech-Emotion-Analyzer

# Install dependencies
pip install -r requirements.txt
