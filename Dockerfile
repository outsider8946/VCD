FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/resources
RUN gdown --id "1i8HKhsUjmE_K-0HODZubXArWPIY-Nn_x" -O app/resources/model.pth
CMD ["streamlit", "run", "app/app.py"]
