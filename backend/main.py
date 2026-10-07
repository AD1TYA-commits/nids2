"""
backend/main.py
===============
Production FastAPI Backend for NIDS SOC Dashboard.
Integrates:
  - Rust JSONL Flow Ingestion (Line-by-line streaming replay or live pipe)
  - Feature Adapter Bridge (backend/adapter.py)
  - Production ML Engine (NIDSDetector)
  - Deterministic Security Risk Engine (RiskEngine)
  - WebSocket Real-Time Event Broadcaster (/ws)
  - SOC Dashboard Frontend Host (/)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure current working directory and subdirectories are on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.adapter import IntegratedNIDS

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nids.backend")

# Initialize FastAPI App
app = FastAPI(
    title="NIDS Real-Time Threat Analysis & SOC Engine",
    description="High-performance NIDS backend combining Subspace Anomaly Detection and Deterministic Risk Rules",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared In-Memory State
MAX_HISTORY_ENTRIES = 500
alert_history: deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY_ENTRIES)

stats = {
    "total_flows": 0,
    "normal_count": 0,
    "suspicious_count": 0,
    "malicious_count": 0,
    "unknown_count": 0,
    "critical_count": 0,
    "total_risk_sum": 0.0,
    "average_risk": 0.0,
}

# Three Explicit Modes Configuration
MODE_LIVE = "live"
MODE_CAPTURED = "captured"
MODE_DEMO = "demo"

MODE_CONFIG = {
    MODE_LIVE: {
        "name": "LIVE CAPTURE",
        "description": "Actual current packets captured via Rust libpcap engine",
        "interface": "wlp4s0",
        "source_file": "None (libpcap stream)",
    },
    MODE_CAPTURED: {
        "name": "CAPTURED TRAFFIC REPLAY",
        "description": "Historical background network capture replay (live_flows.jsonl)",
        "file": ROOT_DIR / "live_flows.jsonl",
        "source_file": "live_flows.jsonl",
    },
    MODE_DEMO: {
        "name": "DEMO / LAB REPLAY",
        "description": "Controlled SOC demonstration feed from benchmark dataset (demo_traffic.jsonl)",
        "file": ROOT_DIR / "demo_traffic.jsonl",
        "source_file": "demo_traffic.jsonl",
    },
}

current_mode: str = os.getenv("FLOW_REPLAY_MODE", MODE_LIVE)
live_interface: str = os.getenv("CAPTURE_INTERFACE", "wlp4s0")
flow_source_file: Path = MODE_CONFIG.get(current_mode, {}).get("file", ROOT_DIR / "demo_traffic.jsonl")

# Global Pipeline Instance
pipeline: Optional[IntegratedNIDS] = None

# Background Replay & Live Process Handles
replay_task: Optional[asyncio.Task] = None
live_bridge_proc: Optional[asyncio.subprocess.Process] = None
is_replaying: bool = False
replay_delay: float = float(os.getenv("FLOW_REPLAY_DELAY", "0.1"))


def reset_stats_and_history():
    """Resets memory queues and aggregate metrics."""
    alert_history.clear()
    stats["total_flows"] = 0
    stats["normal_count"] = 0
    stats["suspicious_count"] = 0
    stats["malicious_count"] = 0
    stats["unknown_count"] = 0
    stats["critical_count"] = 0
    stats["total_risk_sum"] = 0.0
    stats["average_risk"] = 0.0


async def start_live_bridge():
    """Starts the real-time Rust libpcap capture piped to the backend with automatic fallback."""
    global live_bridge_proc
    if live_bridge_proc is not None and live_bridge_proc.returncode is None:
        return

    lib_path = ROOT_DIR / "lib" / "usr" / "lib" / "x86_64-linux-gnu"
    cmd = (
        f"LD_LIBRARY_PATH=\"{lib_path}:$LD_LIBRARY_PATH\" "
        f"./target/release/network-analyzer -i {live_interface} --flow-timeout 1 -f jsonl "
        f"| python3 scripts/pipe_to_backend.py"
    )
    logger.info(f"Launching live packet capture bridge: {cmd}")
    try:
        live_bridge_proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(_monitor_live_bridge(live_bridge_proc))
    except Exception as e:
        logger.error(f"Failed to launch live capture bridge: {e}")
        live_bridge_proc = None
        logger.info("Backend listening for external live flows on /api/classify")


async def _monitor_live_bridge(proc: asyncio.subprocess.Process):
    """Monitors live capture process and keeps backend ready for standalone capture engine."""
    try:
        # Fast check after 0.5s for immediate startup/permission errors
        await asyncio.sleep(0.5)
        if proc.returncode is not None:
            stderr_out = await proc.stderr.read() if proc.stderr else b""
            err_msg = stderr_out.decode(errors="replace").strip()
            logger.info(
                f"Automated live bridge subprocess exited with code {proc.returncode}: {err_msg}. "
                f"Backend is ready and listening for standalone capture engine at /api/classify "
                f"(e.g. 'sudo ./target/release/network-analyzer --interface {live_interface}')."
            )
            return

        # Continuously monitor if the process terminates later
        await proc.wait()
        if current_mode == MODE_LIVE:
            stderr_out = await proc.stderr.read() if proc.stderr else b""
            err_msg = stderr_out.decode(errors="replace").strip()
            logger.info(
                f"Live capture bridge process terminated (code {proc.returncode}): {err_msg}. "
                f"Ready for standalone capture engine at /api/classify."
            )
    except Exception as e:
        logger.error(f"Error monitoring live bridge: {e}")


def _ensure_telemetry_stream():
    """Starts replay worker on live_flows.jsonl only in CAPTURED mode."""
    global is_replaying, flow_source_file, replay_task
    if current_mode == MODE_CAPTURED and not is_replaying:
        fallback_file = ROOT_DIR / "live_flows.jsonl"
        if fallback_file.exists():
            flow_source_file = fallback_file
            logger.info(f"Streaming captured network flow telemetry from {fallback_file}...")
            replay_task = asyncio.create_task(replay_worker())


async def stop_live_bridge():
    """Terminates the live packet capture subprocess."""
    global live_bridge_proc
    if live_bridge_proc is not None:
        logger.info("Stopping live packet capture bridge...")
        try:
            live_bridge_proc.terminate()
            await asyncio.wait_for(live_bridge_proc.wait(), timeout=2.0)
        except Exception:
            try:
                live_bridge_proc.kill()
            except Exception:
                pass
        live_bridge_proc = None


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        payload = json.dumps(message)
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.active_connections.discard(dead)


ws_manager = ConnectionManager()


def record_alert(alert: Dict[str, Any]):
    """Records detection alert in memory and updates running statistical aggregates."""
    alert_history.appendleft(alert)

    stats["total_flows"] += 1
    decision = alert.get("decision", "NORMAL")
    if decision == "NORMAL":
        stats["normal_count"] += 1
    elif decision == "SUSPICIOUS":
        stats["suspicious_count"] += 1
    elif decision == "MALICIOUS":
        stats["malicious_count"] += 1
    elif decision == "UNKNOWN ATTACK":
        stats["unknown_count"] += 1
    elif decision == "CRITICAL":
        stats["critical_count"] += 1

    risk = float(alert.get("risk_score", 0.0))
    stats["total_risk_sum"] += risk
    stats["average_risk"] = round(stats["total_risk_sum"] / max(1, stats["total_flows"]), 2)


async def replay_worker():
    """
    Background worker streaming flow records line-by-line from live_flows.jsonl.
    Does NOT load entire file into memory at once.
    """
    global is_replaying, pipeline
    logger.info(f"Starting flow replay worker from {flow_source_file} (delay={replay_delay}s)...")
    is_replaying = True

    try:
        while is_replaying:
            if not flow_source_file.exists():
                logger.error(f"Flow source file '{flow_source_file}' does not exist.")
                await asyncio.sleep(2.0)
                continue

            with open(flow_source_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not is_replaying:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        if pipeline is not None:
                            alert = pipeline.process_flow(record)
                            record_alert(alert)
                            await ws_manager.broadcast({
                                "type": "flow_alert",
                                "data": alert,
                                "stats": get_stats_payload(),
                            })
                    except Exception as e:
                        logger.error(f"Error processing replay flow record: {e}")

                    await asyncio.sleep(replay_delay)

            # Briefly pause before looping file again
            if is_replaying:
                logger.info("Reached end of flow file, looping replay...")
                await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        logger.info("Replay worker cancelled.")
    finally:
        is_replaying = False
        logger.info("Replay worker stopped.")


def get_stats_payload() -> Dict[str, Any]:
    return {
        "total_flows": stats["total_flows"],
        "normal_count": stats["normal_count"],
        "suspicious_count": stats["suspicious_count"],
        "malicious_count": stats["malicious_count"],
        "unknown_count": stats["unknown_count"],
        "critical_count": stats["critical_count"],
        "average_risk": stats["average_risk"],
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "interface": live_interface if current_mode == MODE_LIVE else None,
        "packet_capture_status": "ONLINE" if (current_mode == MODE_LIVE or is_replaying) else "IDLE",
        "flow_engine_status": "ONLINE",
        "model_status": "ONLINE",
        "risk_engine_status": "ONLINE",
        "source_file": MODE_CONFIG[current_mode].get("source_file", ""),
        "replaying": is_replaying,
        "replay_delay": replay_delay,
    }


# Lifespan Startup / Shutdown
@app.on_event("startup")
async def startup_event():
    global pipeline, replay_task
    logger.info("Initializing Integrated NIDS ML & Risk Engine...")
    pipeline = IntegratedNIDS()
    logger.info("Integrated NIDS pipeline ready.")

    if current_mode == MODE_LIVE:
        await start_live_bridge()
    elif current_mode in (MODE_CAPTURED, MODE_DEMO):
        replay_task = asyncio.create_task(replay_worker())


@app.on_event("shutdown")
async def shutdown_event():
    global is_replaying, replay_task
    is_replaying = False
    if replay_task is not None:
        replay_task.cancel()
    await stop_live_bridge()
    await asyncio.sleep(0.1)


# ==========================================
# REST API Endpoints
# ==========================================

@app.get("/api/status")
async def get_status():
    """System health, loaded model version, active mode, and component status."""
    return {
        "status": "online",
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "monitored_interface": live_interface if current_mode == MODE_LIVE else "N/A (Replay)",
        "packet_capture": "ONLINE" if (current_mode == MODE_LIVE or is_replaying) else "IDLE",
        "flow_engine": "ONLINE",
        "model": "ONLINE",
        "risk_engine": "ONLINE",
        "model_architecture": "Production Multi-Facet Subspace Decomposition (v2.1)",
        "anomaly_detector": "Dual-Subspace PCA + Deep Autoencoder + Isolation Forest",
        "anomaly_threshold": pipeline.detector.threshold if pipeline else 1.3728,
        "total_flows_processed": stats["total_flows"],
        "is_replaying": is_replaying,
        "active_ws_connections": len(ws_manager.active_connections),
    }


@app.get("/api/stats")
async def get_stats():
    """Real-time aggregated detection and risk telemetry."""
    return get_stats_payload()


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Returns recent detection alerts in reverse chronological order."""
    limit = max(1, min(limit, MAX_HISTORY_ENTRIES))
    return list(alert_history)[:limit]


class FlowClassificationRequest(BaseModel):
    flow: Dict[str, Any]


@app.post("/api/classify")
async def classify_flow(req: FlowClassificationRequest):
    """
    Directly classify a single incoming flow JSON record.
    Evaluates ML model and Risk Engine, stores alert, and broadcasts over WebSocket.
    """
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        alert = pipeline.process_flow(req.flow)
        record_alert(alert)
        await ws_manager.broadcast({
            "type": "flow_alert",
            "data": alert,
            "stats": get_stats_payload(),
        })
        return alert
    except Exception as e:
        logger.error(f"Classification failure: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/replay/start")
async def start_replay(delay: Optional[float] = None):
    """Start or adjust the background streaming flow replay."""
    global replay_task, is_replaying, replay_delay
    if delay is not None and delay > 0:
        replay_delay = delay

    if is_replaying:
        return {"status": "already_running", "delay": replay_delay}

    replay_task = asyncio.create_task(replay_worker())
    return {"status": "started", "delay": replay_delay}


@app.post("/api/replay/stop")
async def stop_replay():
    """Pause the background flow replay."""
    global is_replaying, replay_task
    is_replaying = False
    if replay_task is not None:
        replay_task.cancel()
        replay_task = None
    return {"status": "stopped"}


@app.post("/api/replay/mode")
@app.post("/api/mode")
async def set_replay_mode(mode: str):
    """
    Switch between three explicit operational modes:
      - 'live': LIVE CAPTURE (libpcap on monitored interface)
      - 'captured': CAPTURED TRAFFIC REPLAY (live_flows.jsonl)
      - 'demo': DEMO / LAB REPLAY (demo_traffic.jsonl)
    """
    global current_mode, flow_source_file, replay_task, is_replaying
    norm = mode.lower().strip()
    if norm in ("live", "realtime", "raw"):
        new_mode = MODE_LIVE
    elif norm in ("captured", "replay", "real", "background"):
        new_mode = MODE_CAPTURED
    elif norm in ("demo", "lab", "benchmark"):
        new_mode = MODE_DEMO
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Choose 'live' (LIVE CAPTURE), 'captured' (CAPTURED TRAFFIC REPLAY), or 'demo' (DEMO / LAB REPLAY)."
        )

    current_mode = new_mode

    # Stop any running replay worker
    if replay_task is not None:
        replay_task.cancel()
        try:
            await replay_task
        except (asyncio.CancelledError, Exception):
            pass
        replay_task = None
    is_replaying = False

    reset_stats_and_history()

    if current_mode == MODE_LIVE:
        await start_live_bridge()
    else:
        await stop_live_bridge()
        flow_source_file = MODE_CONFIG[current_mode]["file"]
        replay_task = asyncio.create_task(replay_worker())

    payload = {
        "type": "mode_changed",
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "stats": get_stats_payload(),
    }
    await ws_manager.broadcast(payload)
    logger.info(f"Switched operation mode to: {MODE_CONFIG[current_mode]['name']}")
    return {
        "status": "switched",
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "source_file": MODE_CONFIG[current_mode].get("source_file", ""),
        "interface": live_interface if current_mode == MODE_LIVE else None,
        "stats": get_stats_payload(),
    }


@app.post("/api/replay/reset")
async def reset_replay():
    """Reset flow counters, risk averages, and in-memory alert history."""
    reset_stats_and_history()
    payload = {
        "type": "reset",
        "stats": get_stats_payload(),
    }
    await ws_manager.broadcast(payload)
    return {"status": "reset", "stats": get_stats_payload()}



# ==========================================
# WebSocket Endpoint
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket stream emitting real-time detection events and stats."""
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot of stats and recent history
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "stats": get_stats_payload(),
            "recent_alerts": list(alert_history)[:30],
        }))
        while True:
            # Keep socket alive and handle incoming client commands (e.g. ping)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ==========================================
# Static Files & Dashboard UI
# ==========================================

FRONTEND_DIR = ROOT_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>SOC Dashboard not found. Please create frontend/index.html</h2>", status_code=404)
