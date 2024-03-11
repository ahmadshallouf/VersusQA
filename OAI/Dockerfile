FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT sh -c "uvicorn main:app --host=0.0.0.0 --port=8002 --reload"
