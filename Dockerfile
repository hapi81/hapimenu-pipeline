FROM runpod/base:0.6.2-cuda12.2.0

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir runpod requests Pillow

COPY handler.py .

CMD ["python3", "handler.py"]
