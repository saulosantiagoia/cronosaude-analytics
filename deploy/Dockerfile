FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY deploy/requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app/ /app/
ENV CRONOSAUDE_DB=/data/cronosaude.db
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py"]
