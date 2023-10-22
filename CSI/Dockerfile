FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8003

ENTRYPOINT sh -c "uvicorn main:app --host=0.0.0.0 --port=8003 --reload"
