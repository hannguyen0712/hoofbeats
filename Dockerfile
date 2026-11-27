FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p results/sessions

# hosts like Render, Railway, Cloud Run and Fly set $PORT; locally it falls back to 8501
EXPOSE 8501
CMD streamlit run review_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true