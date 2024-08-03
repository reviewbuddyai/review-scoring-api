import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.controllers import router
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure the app directory is in the Python path

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(CustomCORSMiddleware)

app.include_router(router)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Review Scoring API!"}
