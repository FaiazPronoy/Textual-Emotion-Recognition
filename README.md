# Textual Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)

**Naming the emotion in a sentence with LSTM, CNN and hybrid deep learning models.**

| | |
|---|---|
| **Program** | CVIS Hands-on Training Project, BRAC University |
| **Semester** | Summer 2025 |
| **Type** | Individual project |
| **Made by** | Faiaz Abrar Zaman Pronoy |
| **Supervisor** | Dr. Md. Ashraful Alam, CVIS Research Lab |
| **Version** | v1.0, the training submission |

---

## What it does

1. Loads 20,000 English tweets labelled sadness, joy, love, anger, fear or surprise.
2. Cleans the text, stems the words and turns them into padded number sequences.
3. Trains three deep learning models.
4. Combines them with five ensemble methods.
5. Shows word clouds, confusion matrices, classification reports and confidence plots.

## Results

On 1,000 test sentences:

| Model | Accuracy |
|---|---|
| Bi-LSTM | 88.1% |
| 1D CNN | 89.0% |
| Hybrid CNN-LSTM | 89.0% |
| Hard voting ensemble | 89.8% |

## Built with

Python, TensorFlow and Keras, scikit-learn, NLTK, WordCloud, Matplotlib and seaborn.

## Run it

Use a GPU, for example on Kaggle or Colab.

```bash
pip install -r requirements.txt
jupyter notebook textual_emotion_recognition.ipynb
```

## Project structure

```
textual_emotion_recognition.ipynb   the notebook
data/                               training, validation and test sets
report/                             the training report
requirements.txt                    Python packages
```

## Data

The [Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) on Kaggle.

## Author

- **Faiaz Abrar Zaman Pronoy** (ফাইয়াজ আবরার জামান প্রনয়)

## Licence

Copyright 2025-2026 Faiaz Abrar Zaman Pronoy. All rights reserved.
You may read the code and run it to try it out. You may not copy it, hand it in
as your own work, or publish it. See [LICENSE](LICENSE).
