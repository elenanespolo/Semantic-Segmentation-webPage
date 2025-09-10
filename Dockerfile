# Usa Python come base
FROM python:3.10-slim

# Imposta la cartella di lavoro dentro il container
WORKDIR /app

# Installa librerie di sistema minime (compilatori, ecc.)
# Copia i file del progetto nella cartella di lavoro
RUN apt-get update && apt-get install -y git git-lfs \
    && git lfs install \
    && git clone https://github.com/elenanespolo/Semantic-Segmentation-webPage.git /app \
    && rm -rf /var/lib/apt/lists/*

# Ora che requirements.txt è già nel repo clonato, lo puoi installare
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta su cui Flask gira
EXPOSE 5000

# Avvia Flask
CMD ["python", "app.py"]

## docker compose up --build  
