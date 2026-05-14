# load packages
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# load dataset
iris_data = load_iris()

# get class names
class_names = iris_data.target_names


# load data into frame
iris_features = pd.DataFrame(data=iris_data["data"], columns=iris_data["feature_names"])
iris_target = pd.Series(data=iris_data["target"], name="target")
iris_df = pd.concat([iris_features, iris_target], axis=1)

# normalize column
def normalize_column(column):
    new_column = column.lower().strip().split(" ")
    if len(new_column) > 1:
        new_column = "_".join(new_column[:-1])
    else:
        new_column = new_column[0]

    return new_column

iris_df.columns = iris_df.columns.map(normalize_column)

# drop duplicated data
iris_df = iris_df.drop_duplicates(keep="first")

# split dataset
train_df, test_df = train_test_split(iris_df, stratify=iris_df["target"], test_size=0.2, random_state=42)

xtrain = train_df.iloc[:, :-1]
ytrain = train_df.iloc[:, -1]

xtest = test_df.iloc[:, :-1]
ytest = test_df.iloc[:, -1]

pipe = Pipeline(steps=[
    ("scaler", preprocessing.StandardScaler()),
    ("classifier", SVC())
    ])

pipe.fit(xtrain, ytrain)

# evaluate model
test_pred = pipe.predict(xtest)
accuracy = metrics.accuracy_score(ytest, test_pred)
print(f"accuracy: {100 * np.round(accuracy, decimals=3)}%")

# save metrics
eval_metrics = {
    "accuracy": accuracy
    }
with open("scores.json", "w") as file:
    json.dump(eval_metrics, file)

## classification report
print("Displaying classification report")
report = classification_report(ytest, test_pred)
print(report)

# # confusion matrix
# cm = confusion_matrix(ytest, test_pred)
# disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
# disp.plot()
# plt.show()

# serialize model
model_path = "app/model/model.joblib"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(pipe, model_path)


# save test data
test_df.reset_index(drop=True).to_csv("test_data.csv", index=False)
