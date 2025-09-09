# Usa Python come base
FROM python:3.10-slim

# Imposta la cartella di lavoro dentro il container
WORKDIR /app

# Installa librerie di sistema minime (compilatori, ecc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia i file del progetto nella cartella di lavoro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN apt-get update && apt-get install -y git git-lfs \
    && git lfs install \
    && git lfs pull

# Espone la porta su cui Flask gira
EXPOSE 5000

# Avvia Flask
CMD ["python", "app.py"]
