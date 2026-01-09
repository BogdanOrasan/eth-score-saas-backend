from fastapi import FastAPI
from .docs_routes import router as docs_router

app = FastAPI()

app.include_router(docs_router)


@app.get("/health")
def health():
    return {"status": "ok"}
