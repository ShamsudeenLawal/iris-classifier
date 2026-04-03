import os
import joblib

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# load model
# model_path = os.path.join("model", "model.joblib")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.joblib")
model = joblib.load(model_path)

class_names = ['setosa', 'versicolor', 'virginica']

# define item class
class Item(BaseModel):
    sepal_length : float | int
    sepal_width : float | int
    petal_length : float | int
    petal_width : float | int


# initialize app
app = FastAPI()

# define endpoints
@app.get("/")
def home():
    return {"status": "Congratulations, your server is now live."}

@app.post("/predict")
def get_prediction(item: Item):
    features = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width], dtype=np.float32)
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": class_names[prediction]}

if __name__ == "__main__":
    uvicorn.run("main:app", port=80, reload=True)