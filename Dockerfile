FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*  # Clean up

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y build-essential python3-dev && apt-get autoremove -y

COPY . /app/server

WORKDIR /app/server

EXPOSE 8080

CMD ["python", "main.py"]