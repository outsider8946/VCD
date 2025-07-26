FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/resources
RUN gdown --id "1EoNZ7OHkj3bCv0EMfYJFI44y1GMP7Fxt" -O app/resources/model.pth
CMD ["streamlit", "run", "app/app.py"]
