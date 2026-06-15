FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Argument pozwalający zbudować wersję 1.0 (app.py) albo 2.0 (app_v2.py)
ARG APP_FILE=app.py
COPY ${APP_FILE} app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
