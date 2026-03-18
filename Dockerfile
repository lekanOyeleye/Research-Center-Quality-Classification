FROM python:3.12-slim

# Step 2 — Environment variables Do not create .pyc bytecode files 
ENV PYTHONDONTWRITEBYTECODE=1   
ENV PYTHONUNBUFFERED=1          

# Step 3 — Set working directory inside the container 
WORKDIR /app 

# Step 4 — Copy requirements FIRST (separate layer for Docker caching).This Install system deps (if needed) and Python deps
COPY requirements.txt . 

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc-dev \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y gcc libc-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

 # Copy app code and model file(s)
COPY . .

# Train the Model and store in the model/artifacts directory
RUN python3 train.py

# Expose the port your app runs on
EXPOSE 8000

# Run the app application using uvicorn server on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]