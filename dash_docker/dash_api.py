import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import logging
import asyncio
import hashlib
import sys
import astrodash

logger = logging.getLogger("startup")

class Params(BaseModel):
    spectrum: str | None = ''
    redshift: float | None = 0.0
    classify_host: bool | None = False
    known_z: bool | None = True
    smooth: int | None = 6
    rlap: bool | None = False
    output_dir: str

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    dash_name = 'astrodash'
    if dash_name in sys.modules:
        logger.info("Startup successful")
    else:
        logger.info(f"Error: {dash_name} not found:")

@app.get("/health")
async def health():

    return {
            "status": "ok",
            }

_running_requests = {}
_running_lock = asyncio.Lock()

def _hash_params(params: dict):
    key_str = f"{params['spectrum']}-{params['redshift']}-{params['smooth']}"
    return hashlib.md5(key_str.encode()).hexdigest()

@app.post("/dash_params/")
async def run_dash(params: Params):

    key = _hash_params(params.dict())

    async with _running_lock:
        if key in _running_requests:
            return await _running_requests[key]

        task = asyncio.create_task(_run_dash_task(params))

        _running_requests[key] = task

        try:
            result = await task
            return result
        finally:
            _running_requests.pop(key, None)

async def _run_dash_task(params: Params):
    params = params.dict()

    if not os.path.abspath(params['output_dir']).startwith('/dash_api_runs'):
        raise ValueError('Invalid output directory')

    classification = astrodash.Classify([params['spectrum']], params['redshift'],
                                        params['classify_host'], params['known_z'].
                                        params['smooth'], params['rlap'])

    classification.list_best_matches(n=100,
                                     saveFilename=f"{params['output_dir']}Dash_matches.txt")

    return {"success": True, "data": {"file_path": f"{params['output_dir']}/"}} #TODO set output dir
