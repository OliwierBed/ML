from fastapi import FastAPI
from api.routers.ml import router as ml_router

app = FastAPI(title="ML API")

app.include_router(ml_router, prefix="/ml")
