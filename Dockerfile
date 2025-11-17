# 1. Gunakan Python rasmi
FROM python:3.13-slim

# 2. Install dependencies sistem yang diperlukan (contoh zbar)
RUN apt-get update && \
    apt-get install -y zbar-tools libzbar0 && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory dalam container
WORKDIR /app

# 4. Salin semua files project ke dalam container
COPY . /app

# 5. Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6. Set environment variable untuk Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# 7. Expose port 5000
EXPOSE 5000

# 8. Jalankan app dengan gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
