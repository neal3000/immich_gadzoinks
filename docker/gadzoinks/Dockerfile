# gadzoinks/Dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi uvicorn psycopg2-binary requests
COPY main.py .
COPY MetadataExtractor.py .
EXPOSE 2289
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2289"]
