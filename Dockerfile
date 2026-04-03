# collect python base image
FROM python:3.11-slim
# set working directory
WORKDIR /iris-classifier
# copy requirements file
COPY requirements.txt /iris-classifier/requirements.txt
# upgrade, install and remove requirements file (for lightweight image, requirements.txt is a small file anyway)
RUN pip install --no-cache-dir --upgrade -r /iris-classifier/requirements.txt
# copy fastapi deployment app directory
COPY app /iris-classifier/app
# command to execute on container startup
CMD [ "fastapi", "app/main.py", "--port", "80" ]