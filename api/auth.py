from fastapi import Header, HTTPException
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

def validate_api_key(authorization: str = Header(...)):
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token not in API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    except Exception:
        raise HTTPException(status_code=401, detail="Authorization header is required")