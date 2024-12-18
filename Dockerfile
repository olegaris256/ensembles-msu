FROM python:3.12.5

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt
