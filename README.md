# 🌸 Iris Flower Classification (Production-Ready ML System)

A complete end-to-end machine learning system for classifying Iris flower species using classical ML algorithms. The project demonstrates a production-style ML pipeline including data preprocessing, model selection via cross-validation, API deployment using FastAPI, and containerization with Docker.

---

## 📌 Project Overview

This project implements a **reproducible machine learning pipeline** that includes:

- Data ingestion and validation
- Data preprocessing and standardization
- Model benchmarking using cross-validation
- Best model selection based on performance
- Pipeline serialization for reproducibility
- REST API deployment using FastAPI
- Dockerized deployment for portability

---

## 🧠 Problem Statement

Given measurements of iris flowers (sepal length, sepal width, petal length, petal width), the goal is to classify each sample into one of three species:

- Setosa
- Versicolor
- Virginica

---

## 📊 Dataset

- Source: `scikit-learn` built-in Iris dataset
- Features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- Target:
  - 0 → Setosa
  - 1 → Versicolor
  - 2 → Virginica

---

## 🧪 Machine Learning Pipeline

### Workflow

1. Load dataset (scikit-learn Iris dataset)
2. Data cleaning
   - Removed duplicate records
3. Exploratory Data Analysis (EDA)
4. Feature scaling (Standardization)
5. Model benchmarking using cross-validation:
   - Logistic Regression
   - Ridge Classifier
   - SVC
   - KNN
   - AdaBoost
   - Decision Tree
   - Random Forest
   - SGD Classifier
   - MLP Classifier

6. Model evaluation metric:
   - Accuracy

7. Best model:
   - ✅ Support Vector Classifier (SVC)
   - ~97% mean validation accuracy

8. Final pipeline:
   - Scaler + Model combined into a single reproducible pipeline

---

## 🚀 API Deployment (FastAPI)

### Start Server

```bash
python app/main.py
```

Server runs on:

`http://localhost:8080`

### 📡 API Endpoint

POST `/predict`

### Predict iris species from input features.

Request Body:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```json
{
  "prediction": "setosa"
}
```

## 🧪 Testing the API

A client script is provided for automated testing:

```bash
python client.py
```

- Uses unseen samples from the Iris dataset
- Sends requests to FastAPI endpoint
- Validates model predictions

## 🐳 Docker Deployment

- Build Image

```bash
docker build -t iris-classifier .
```

- Run Container

```bash
docker run -d --name iris-classifier-container -p 8080:8080 iris-classifier
```

## Access API

http://localhost:8080
