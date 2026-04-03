# Iris Flower Classification

Build IMAGE: docker build -t iris-classifier .
RUN CONTAINER: docker run -d --name iris-classifier-container -p 80:80 iris-classifier
