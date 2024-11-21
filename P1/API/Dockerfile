FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install necessary tools
RUN apt-get update && apt-get install -y docker.io
RUN apt-get update && apt-get install -y iputils-ping

# Add a new user and assign it to the docker group
RUN groupadd docker && useradd -m -g docker dockeruser

# Switch to the new user
USER dockeruser

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "practice1:app", "--host", "0.0.0.0", "--port", "80", "--reload"]