FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libsm6

COPY . .

RUN mkdir -p app/resources
RUN gdown --id "1EoNZ7OHkj3bCv0EMfYJFI44y1GMP7Fxt" --output app/resources/model.pth

CMD ["streamlit", "run", "app/streamlit_app.py"]
