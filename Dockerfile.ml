FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "api.ml_main:app", "--host", "0.0.0.0", "--port", "8001"]
