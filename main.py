# main.py
from fastapi import FastAPI
from api.routers import test

app = FastAPI()

app.include_router(test.router, prefix="/test")

@app.get("/")
def root():
    return {"message": "API działa poprawnie!"}
