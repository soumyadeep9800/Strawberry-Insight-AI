from fastapi import FastAPI
from routes.advisory import router as advisory_router

app = FastAPI(
    title="Strawberry Disease Advisory Service",
    description="AI-powered advisory service for hydroponic strawberry diseases",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "Strawberry Disease Advisory Service",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


app.include_router(advisory_router)