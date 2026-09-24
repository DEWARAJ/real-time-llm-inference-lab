#!/usr/bin/env python3
"""Run the SLO-aware gateway with uvicorn."""

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "inference_lab.gateway_app:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8080")),
    )
