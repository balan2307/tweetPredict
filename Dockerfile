FROM python:3.7-slim-buster

COPY predict.py .

COPY . /app

# Copy the dataset file into the container at /app/data
COPY tweet.csv /app/data/

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "predict.py"]
