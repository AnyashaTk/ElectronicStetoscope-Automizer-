FROM python:3.10

WORKDIR /app

COPY app /app

RUN pip install flask

EXPOSE 5050

CMD ["python", "main.py"]
