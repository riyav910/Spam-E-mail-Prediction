# Spam Mail Prediction

This project aims to classify emails as spam or ham using a machine learning model, specifically Logistic Regression. The model processes email data and predicts whether a given message is spam or genuine (ham), providing users with an efficient way to filter unwanted emails.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Workflow](#workflow)
  - [1. Importing Dependencies](#1-importing-dependencies)
  - [2. Data Collection & Pre-processing](#2-data-collection--pre-processing)
  - [3. Label Encoding](#3-label-encoding)
  - [4. Splitting Data](#4-splitting-data)
  - [5. Feature Extraction](#5-feature-extraction)
  - [6. Model Training](#6-model-training)
  - [7. Evaluation](#7-evaluation)
  - [8. Prediction](#8-prediction)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Project Overview

This project uses a logistic regression model to predict whether an email is spam based on its content. Using the `TfidfVectorizer`, we convert the email text into numerical data that the model can analyze. This project is created on Google Colab and includes data pre-processing, training, and evaluation.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

## Setup Instructions

1. Clone this repository.
2. Upload your dataset to Google Colab and load it using `pd.read_csv()`.
3. Run the notebook to train the model and test predictions.

## Workflow

#### 1. Importing Dependencies

#### 2. Data Collection & Pre-processing
- Load the email dataset from a CSV file.
- Replace any null values with empty strings to ensure consistency.

#### 3. Label Encoding
- Assign numeric labels to the categories: `0` for spam and `1` for ham.

#### 4. Splitting Data
- Separate the dataset into features (`X`) and labels (`Y`).
- Split the dataset into training and test sets for model evaluation.

#### 5. Feature Extraction
- Use `TfidfVectorizer` to transform text data into feature vectors, focusing on word frequency and relevance.

#### 6. Model Training
- Train a logistic regression model using the transformed training data for spam detection.

#### 7. Evaluation
- Measure model accuracy on training and test data to assess performance.

#### 8. Prediction
- Test the model by making predictions on sample email messages to classify them as spam or ham.


## Results
- **Training Accuracy**: 96.7%
- **Test Accuracy**: 96.6%

## Future Enhancements
- Implement additional NLP preprocessing for improved accuracy.
- Experiment with different classification algorithms, such as SVM or Naive Bayes, to explore other modeling approaches.

