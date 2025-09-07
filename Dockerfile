# Usa Python come base
FROM python:3.10-slim

# Imposta la cartella di lavoro dentro il container
WORKDIR /app

# Copia i file del progetto nella cartella di lavoro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Espone la porta su cui Flask gira
EXPOSE 5000

# Avvia Flask
CMD ["python", "app.py"]
