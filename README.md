# Facial Emotion Recognition

A **CNN with L2 regularization** that classifies facial expressions into three emotions: **angry**, **happy** and **sad**. The dataset was collected by **web scraping with Python** and cleaned before training. Deep Learning course group project (Group 1).

## Dataset

- About 1,370 images were scraped into three classes (`angry_face`, `happy_face`, `sad_face`, roughly balanced).
- After removing duplicate (hash-based) and corrupt images, **1,252 images** were used for modeling.
- **The images are not included in this repository** for privacy reasons, because they show real people's faces. Image previews were also removed from the notebook outputs.

## Pipeline

1. **Preprocessing**: load images into a DataFrame, remove duplicates and corrupt files
2. **EDA**: image dimensions, class distribution, RGB color distribution, aspect ratio, brightness, file size
3. **Image enhancement**: grayscale conversion and **CLAHE** (contrast-limited adaptive histogram equalization)
4. **Resizing** to 48×48 grayscale, label encoding, train/test split
5. **Modeling**: CNN with Conv2D blocks, **Batch Normalization**, **Dropout** and **L2 weight regularization** to reduce overfitting
6. **Training & evaluation**: accuracy/loss curves, classification report, confusion matrix
7. **Model export** to `.h5`

## Repository structure

```
notebooks/facial_emotion_recognition.ipynb   # full workflow (text outputs kept, image outputs removed)
requirements.txt
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/facial_emotion_recognition.ipynb
```

Place your own images in one folder per class (`angry_face/`, `happy_face/`, `sad_face/`) and update the dataset path in the first cells. The notebook was developed in Google Colab; its commentary is partly in Indonesian.

## Tech stack

Python · TensorFlow/Keras · OpenCV · scikit-learn · NumPy · pandas · Pillow · Matplotlib · Seaborn
