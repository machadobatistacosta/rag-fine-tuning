"""Test bootstrap for running the FastAPI app during the test suite."""
from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Optional

import uvicorn


_SERVER: Optional[uvicorn.Server] = None
_SERVER_THREAD: Optional[threading.Thread] = None


def _ensure_server_started() -> None:
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        return

    port = int(os.getenv("RAG_API_PORT", "8000"))
    config = uvicorn.Config("backend.main:app", host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    thread = threading.Thread(target=_run_server, name="uvicorn-test-server", daemon=True)
    thread.start()

    # Aguarda o servidor ficar disponivel antes de prosseguir com os testes
    for _ in range(200):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Nao foi possivel iniciar o servidor de testes do FastAPI.")

    _SERVER = server
    _SERVER_THREAD = thread

    atexit.register(_shutdown_server)


def _shutdown_server() -> None:
    if _SERVER is None:
        return
    _SERVER.should_exit = True
    if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
        _SERVER_THREAD.join(timeout=1)


_ensure_server_started()
