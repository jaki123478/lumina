FROM python:3.11-slim
WORKDIR /app
COPY hf_complete_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lumina_onnx_runtime.py .
COPY hf_complete_server/app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
