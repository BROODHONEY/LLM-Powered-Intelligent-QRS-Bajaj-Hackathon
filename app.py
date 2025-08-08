import os
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
from typing import List

from api.auth import validate_api_key
from api.hackrx_pipeline import ingest_link_and_build, answer_queries
from query_parser import HackRxRequest, HackRxResponse

app = FastAPI(title="HackRx LLM Query Retriever")

@app.post("/hackrx/run", response_model=HackRxResponse)
def run_hackrx(
    request: HackRxRequest,
    fastapi_request: Request,
    _: None = Depends(validate_api_key)
):
    print("[INFO] Ingesting documents and building index...")
    ingest_link_and_build(request.documents)
    print("[INFO] Ingesting documents and building index complete.")
    print("[INFO] Answering questions...")
    answers = answer_queries(request.questions) 
    print("[INFO] Answering questions complete.")
    return HackRxResponse(answers=answers)