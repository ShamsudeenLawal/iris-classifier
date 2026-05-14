import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# load model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.joblib")
model = joblib.load(model_path)

class_names = ['setosa', 'versicolor', 'virginica']

# define item class
class Iris(BaseModel):
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
def get_prediction(iris: Iris):
    x = np.array([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width], dtype=np.float32)
    x = x.reshape(1, -1)
    prediction = model.predict(x)[0]
    return {"prediction": class_names[prediction]}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8080)