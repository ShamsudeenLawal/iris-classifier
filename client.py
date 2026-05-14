
import json
import requests
import numpy as np
import pandas as pd

# load data
df = pd.read_csv('test_data.csv')
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

class_names = ['setosa', 'versicolor', 'virginica']

# convert dataframe to json
json_data_string = features.to_json(orient="records")
json_data = json.loads(json_data_string)

## get prediction
# index = 0
# url = "http://127.0.0.1:8080/predict"
# resp = requests.post(url, json=json_data[index])

# response_dict = resp.json()
# prediction = response_dict["prediction"]

# # print predictions and true label
# true_label = labels.iat[index]
# print(f"Prediction: {prediction}\nTrue label: {class_names[true_label]}")

for i in range(10):
    # index = 0
    url = "http://127.0.0.1:8080/predict"
    resp = requests.post(url, json=json_data[i])

    response_dict = resp.json()
    prediction = response_dict["prediction"]

    # print predictions and true label
    true_label = labels.iat[i]
    print(f"Prediction: {prediction}\nTrue label: {class_names[true_label]}", end="\n---\n")
