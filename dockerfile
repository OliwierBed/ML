FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN touch backtest/__init__.py && touch api/__init__.py

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000
EXPOSE 8501

ENV PYTHONPATH=/app

CMD python data-pipelines/fetchers/download_data.py && \
    python data-pipelines/preprocessors/tech_indicators.py && \
    python -m backtest.runner_batch && \
    python -m backtest.score_strategies && \
    python -m backtest.ensemble && \
    python prepare_full_ensemble.py && \
    uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
