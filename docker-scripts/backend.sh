#!/bin/sh
uvicorn ensembles.backend.app:app --host 0.0.0.0 --port 8000 --reload