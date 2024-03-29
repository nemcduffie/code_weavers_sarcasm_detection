FROM python:3.11-slim

WORKDIR /src
ENV PYTHONUNBUFFERED 1
ENV PORT 80

COPY requirements.txt /src/requirements.txt

RUN apt-get update
RUN apt-get install sudo -y build-essential \
    python3-pip \
    python3-dev
RUN pip install --no-cache-dir -r requirements.txt

COPY . /src
COPY ./data/ /src/data/

RUN apt-get clean
