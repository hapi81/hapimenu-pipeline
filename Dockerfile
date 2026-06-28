FROM python:3.11-slim

WORKDIR /app

RUN pip3 install --no-cache-dir runpod

COPY handler.py .

CMD ["python3", "handler.py"]
