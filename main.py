"""
Autonomous Predictive Learning Node
Complete FastAPI application with online Euclidean clustering, Markov state transitions,
epsilon-greedy Q-learning, background consolidation, and live dark-mode dashboard.
"""

import os
import json
import math
import time
import random
import asyncio
import threading
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try importing psycopg2 for PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# ==========================================
# Database Configuration & Auto-Detection
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_TYPE = "sqlite"
DB_NAME = "mind.db"

if DATABASE_URL and PSYCOPG2_AVAILABLE:
    # Adjust postgres:// to postgresql:// for psycopg2 if necessary
    pg_url = DATABASE_URL
    if pg_url.startswith("postgres://"):
        pg_url = "postgresql://" + pg_url[11:]
    try:
        test_conn = psycopg2.connect(pg_url, connect_timeout=5)
        test_conn.close()
        DB_TYPE = "postgres"
        DATABASE_URL = pg_url
        print(f"[*] Successfully connected to PostgreSQL via DATABASE_URL")
    except Exception as e:
        print(f"[!] Warning: PostgreSQL connection failed ({e}). Falling back to SQLite.")
        DB_TYPE = "sqlite"
else:
    DB_TYPE = "sqlite"
    print(f"[*] Operating in SQLite mode: {DB_NAME}")

db_lock = threading.Lock()

def get_connection():
    if DB_TYPE == "postgres":
        return psycopg2.connect(DATABASE_URL)
    else:
        conn = sqlite3.connect(DB_NAME, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

def init_db():
    with db_lock:
        conn = get_connection()
        try:
            if DB_TYPE == "postgres":
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS clusters (
                            state_id VARCHAR(64) PRIMARY KEY,
                            centroid TEXT NOT NULL,
                            vector_length INTEGER NOT NULL,
                            sample_count INTEGER NOT NULL DEFAULT 1,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE IF NOT EXISTS q_table (
                            state_id VARCHAR(64) NOT NULL,
                            action VARCHAR(128) NOT NULL,
                            q_value DOUBLE PRECISION DEFAULT 0.0,
                            visit_count INTEGER DEFAULT 0,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (state_id, action)
                        );
                        CREATE TABLE IF NOT EXISTS markov_transitions (
                            from_state VARCHAR(64) NOT NULL,
                            to_state VARCHAR(64) NOT NULL,
                            transition_count INTEGER DEFAULT 1,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (from_state, to_state)
                        );
                        CREATE TABLE IF NOT EXISTS experiences (
                            id SERIAL PRIMARY KEY,
                            agent_id VARCHAR(128) NOT NULL,
                            modality VARCHAR(128) NOT NULL,
                            value DOUBLE PRECISION NOT NULL,
                            state_id VARCHAR(64) NOT NULL,
                            predicted_next_state VARCHAR(64) NOT NULL,
                            action VARCHAR(128) NOT NULL,
                            actual_reward DOUBLE PRECISION DEFAULT NULL,
                            has_feedback INTEGER DEFAULT 0,
                            consolidated INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE IF NOT EXISTS agent_states (
                            agent_id VARCHAR(128) PRIMARY KEY,
                            last_state_id VARCHAR(64) NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE IF NOT EXISTS history_snapshots (
                            id SERIAL PRIMARY KEY,
                            rolling_reward DOUBLE PRECISION NOT NULL,
                            discovered_states_count INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()
            else:
                cur = conn.cursor()
                cur.executescript("""
                    CREATE TABLE IF NOT EXISTS clusters (
                        state_id TEXT PRIMARY KEY,
                        centroid TEXT NOT NULL,
                        vector_length INTEGER NOT NULL,
                        sample_count INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS q_table (
                        state_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        q_value REAL DEFAULT 0.0,
                        visit_count INTEGER DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (state_id, action)
                    );
                    CREATE TABLE IF NOT EXISTS markov_transitions (
                        from_state TEXT NOT NULL,
                        to_state TEXT NOT NULL,
                        transition_count INTEGER DEFAULT 1,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (from_state, to_state)
                    );
                    CREATE TABLE IF NOT EXISTS experiences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL,
                        modality TEXT NOT NULL,
                        value REAL NOT NULL,
                        state_id TEXT NOT NULL,
                        predicted_next_state TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actual_reward REAL DEFAULT NULL,
                        has_feedback INTEGER DEFAULT 0,
                        consolidated INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS agent_states (
                        agent_id TEXT PRIMARY KEY,
                        last_state_id TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS history_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rolling_reward REAL NOT NULL,
                        discovered_states_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
        finally:
            conn.close()

def db_query(query: str, params: tuple = (), one: bool = False) -> Any:
    with db_lock:
        conn = get_connection()
        try:
            if DB_TYPE == "postgres":
                query_pg = query.replace("?", "%s")
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query_pg, params)
                    if cur.description:
                        rows = [dict(r) for r in cur.fetchall()]
                        return (rows[0] if rows else None) if one else rows
                    conn.commit()
                    return None
            else:
                cur = conn.cursor()
                cur.execute(query, params)
                if cur.description:
                    rows = [dict(r) for r in cur.fetchall()]
                    return (rows[0] if rows else None) if one else rows
                conn.commit()
                return None
        finally:
            conn.close()

def db_execute(query: str, params: tuple = ()) -> None:
    with db_lock:
        conn = get_connection()
        try:
            if DB_TYPE == "postgres":
                query_pg = query.replace("?", "%s")
                with conn.cursor() as cur:
                    cur.execute(query_pg, params)
                    conn.commit()
            else:
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
        finally:
            conn.close()

def db_insert_experience(agent_id: str, modality: str, value: float, state_id: str,
                         predicted_next_state: str, action: str) -> int:
    with db_lock:
        conn = get_connection()
        try:
            if DB_TYPE == "postgres":
                sql = """
                    INSERT INTO experiences (agent_id, modality, value, state_id, predicted_next_state, action, actual_reward, has_feedback, consolidated)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, 0, 0)
                    RETURNING id;
                """
                with conn.cursor() as cur:
                    cur.execute(sql, (agent_id, modality, value, state_id, predicted_next_state, action))
                    new_id = cur.fetchone()[0]
                    conn.commit()
                    return new_id
            else:
                sql = """
                    INSERT INTO experiences (agent_id, modality, value, state_id, predicted_next_state, action, actual_reward, has_feedback, consolidated)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0);
                """
                cur = conn.cursor()
                cur.execute(sql, (agent_id, modality, value, state_id, predicted_next_state, action))
                new_id = cur.lastrowid
                conn.commit()
                return new_id
        finally:
            conn.close()


# ==========================================
# Core Predictive Learning Logic
# ==========================================
EUCLIDEAN_THRESHOLD = 0.45
EPSILON = 0.15          # Exploration rate
LEARNING_RATE = 0.1     # Alpha
DISCOUNT_FACTOR = 0.9   # Gamma

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def find_or_create_cluster(context_vector: List[float]) -> tuple[str, bool, Optional[float]]:
    """
    Online Euclidean clustering with 0.45 threshold.
    Groups incoming vectors of any dimension into STATE_01, STATE_02, etc.
    Updates the centroid on match using incremental online averaging.
    """
    dim = len(context_vector)
    clusters = db_query("SELECT state_id, centroid, vector_length, sample_count FROM clusters WHERE vector_length = ?", (dim,))

    best_match = None
    min_dist = float("inf")

    if clusters:
        for c in clusters:
            try:
                centroid = json.loads(c["centroid"])
                if len(centroid) == dim:
                    dist = euclidean_distance(context_vector, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = c
            except Exception:
                continue

    if best_match is not None and min_dist <= EUCLIDEAN_THRESHOLD:
        # Match found: update centroid incrementally
        state_id = best_match["state_id"]
        old_centroid = json.loads(best_match["centroid"])
        n = best_match["sample_count"] + 1
        new_centroid = [old_c + (x - old_c) / n for old_c, x in zip(old_centroid, context_vector)]

        db_execute(
            "UPDATE clusters SET centroid = ?, sample_count = ?, updated_at = CURRENT_TIMESTAMP WHERE state_id = ?",
            (json.dumps(new_centroid), n, state_id)
        )
        return state_id, False, min_dist

    # No match within threshold or dimension unseen: discover a new state
    all_clusters_count_row = db_query("SELECT COUNT(*) as count FROM clusters", one=True)
    count = all_clusters_count_row["count"] if all_clusters_count_row else 0
    new_state_id = f"STATE_{count + 1:02d}"

    db_execute(
        "INSERT INTO clusters (state_id, centroid, vector_length, sample_count, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (new_state_id, json.dumps(context_vector), dim, 1)
    )
    return new_state_id, True, (min_dist if min_dist != float("inf") else 0.0)

def record_markov_and_predict(agent_id: str, current_state: str) -> str:
    """
    Records agent transition (last_state -> current_state) and
    predicts the next state from current_state based on transition counts.
    """
    agent_record = db_query("SELECT last_state_id FROM agent_states WHERE agent_id = ?", (agent_id,), one=True)

    if agent_record and agent_record["last_state_id"]:
        prev_state = agent_record["last_state_id"]
        # Increment Markov transition
        db_execute("""
            INSERT INTO markov_transitions (from_state, to_state, transition_count, updated_at)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (from_state, to_state) DO UPDATE SET
                transition_count = markov_transitions.transition_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (prev_state, current_state))

    # Update agent's last known state
    db_execute("""
        INSERT INTO agent_states (agent_id, last_state_id, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (agent_id) DO UPDATE SET
            last_state_id = excluded.last_state_id,
            updated_at = CURRENT_TIMESTAMP
    """, (agent_id, current_state))

    # Predict next state from current_state
    transitions = db_query(
        "SELECT to_state, transition_count FROM markov_transitions WHERE from_state = ? ORDER BY transition_count DESC",
        (current_state,)
    )

    if transitions and len(transitions) > 0:
        return transitions[0]["to_state"]
    else:
        # Default self-prediction if no transitions recorded yet
        return current_state

def choose_action(state_id: str, available_actions: List[str]) -> tuple[str, bool]:
    """
    Picks an action using epsilon-greedy Q-learning.
    """
    if not available_actions:
        available_actions = ["MAINTAIN", "SCALE_UP", "THROTTLE", "REBALANCE"]

    # Exploration
    if random.random() < EPSILON:
        return random.choice(available_actions), True

    # Exploitation: read Q-values
    q_rows = db_query("SELECT action, q_value FROM q_table WHERE state_id = ?", (state_id,))
    q_map = {row["action"]: float(row["q_value"]) for row in (q_rows or [])}

    best_q = -float("inf")
    best_actions = []

    for action in available_actions:
        val = q_map.get(action, 0.0)
        if val > best_q:
            best_q = val
            best_actions = [action]
        elif math.isclose(val, best_q, abs_tol=1e-5):
            best_actions.append(action)

    if best_actions:
        return random.choice(best_actions), False
    return random.choice(available_actions), False


# ==========================================
# Background Consolidation Worker (5s cycle)
# ==========================================
async def consolidation_background_worker():
    """
    Runs every 5 seconds:
    1. Replays unconsolidated experiences to update Q-values.
    2. Prunes experiences table over 1,000 rows.
    3. Records periodic history snapshot.
    """
    print("[*] Autonomous consolidation worker started (5s interval).")
    while True:
        try:
            await asyncio.sleep(5)
            # 1. Replay unconsolidated experiences
            unconsolidated = db_query("""
                SELECT id, state_id, action, predicted_next_state, actual_reward, has_feedback
                FROM experiences
                WHERE consolidated = 0
                ORDER BY id ASC
                LIMIT 100
            """)

            if unconsolidated:
                for exp in unconsolidated:
                    s = exp["state_id"]
                    a = exp["action"]
                    s_next = exp["predicted_next_state"]
                    reward = float(exp["actual_reward"]) if exp["actual_reward"] is not None else 0.0

                    # Find current Q(s, a)
                    current_q_row = db_query(
                        "SELECT q_value, visit_count FROM q_table WHERE state_id = ? AND action = ?",
                        (s, a), one=True
                    )
                    curr_q = float(current_q_row["q_value"]) if current_q_row else 0.0
                    visits = int(current_q_row["visit_count"]) if current_q_row else 0

                    # Find max_a' Q(s', a')
                    next_q_rows = db_query("SELECT q_value FROM q_table WHERE state_id = ?", (s_next,))
                    max_next_q = max([float(r["q_value"]) for r in next_q_rows], default=0.0) if next_q_rows else 0.0

                    # Bellman update
                    target = reward + DISCOUNT_FACTOR * max_next_q
                    new_q = curr_q + LEARNING_RATE * (target - curr_q)

                    # Upsert to q_table
                    db_execute("""
                        INSERT INTO q_table (state_id, action, q_value, visit_count, updated_at)
                        VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                        ON CONFLICT (state_id, action) DO UPDATE SET
                            q_value = ?,
                            visit_count = q_table.visit_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                    """, (s, a, new_q, new_q))

                    # Mark experience consolidated
                    db_execute("UPDATE experiences SET consolidated = 1 WHERE id = ?", (exp["id"],))

            # 2. Prune logs over 1,000 rows
            count_row = db_query("SELECT COUNT(*) as count FROM experiences", one=True)
            total_exp = count_row["count"] if count_row else 0
            if total_exp > 1000:
                cutoff_row = db_query("SELECT id FROM experiences ORDER BY id DESC LIMIT 1 OFFSET 999", one=True)
                if cutoff_row and cutoff_row.get("id"):
                    db_execute("DELETE FROM experiences WHERE id < ?", (cutoff_row["id"],))

            # 3. Take snapshot for rolling history
            recent_feedback = db_query("""
                SELECT actual_reward FROM experiences
                WHERE actual_reward IS NOT NULL
                ORDER BY id DESC LIMIT 30
            """)
            if recent_feedback:
                rewards = [float(r["actual_reward"]) for r in recent_feedback]
                rolling_avg_reward = sum(rewards) / len(rewards)
            else:
                rolling_avg_reward = 0.0

            clusters_count_row = db_query("SELECT COUNT(*) as count FROM clusters", one=True)
            total_clusters = clusters_count_row["count"] if clusters_count_row else 0

            db_execute(
                "INSERT INTO history_snapshots (rolling_reward, discovered_states_count, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (rolling_avg_reward, total_clusters)
            )

            # Keep snapshots table bounded (last 60)
            db_execute("""
                DELETE FROM history_snapshots WHERE id NOT IN (
                    SELECT id FROM history_snapshots ORDER BY id DESC LIMIT 60
                )
            """)

        except asyncio.CancelledError:
            break
        except Exception as err:
            print(f"[!] Error in background consolidation task: {err}")


# ==========================================
# FastAPI Lifespan & App Setup
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    bg_task = asyncio.create_task(consolidation_background_worker())
    yield
    bg_task.cancel()
    try:
        await bg_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Autonomous Predictive Learning Node",
    description="Online Euclidean clustering, Markov transitions, and Q-learning autonomous node.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS enabled for all origins so external web apps can send POST requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Request Models
# ==========================================
class SignalRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    modality: str = Field(..., description="Sensor or signal modality name")
    value: float = Field(..., description="Scalar signal reading")
    context_vector: List[float] = Field(..., description="Context vector of arbitrary dimension")
    available_actions: Optional[List[str]] = Field(
        default_factory=lambda: ["MAINTAIN", "SCALE_UP", "THROTTLE", "REBALANCE"],
        description="List of available actions"
    )

class FeedbackRequest(BaseModel):
    event_id: int = Field(..., description="Identifier of the recorded experience event")
    actual_reward: float = Field(..., description="Reward signal (e.g., -1.0 to +1.0 or real-valued)")


# ==========================================
# API Endpoints
# ==========================================

@app.post("/api/signal")
async def post_signal(req: SignalRequest):
    """
    Receives an incoming signal and context vector:
    1. Discovers or assigns state via online Euclidean clustering (threshold 0.45).
    2. Records Markov state transition and predicts next state.
    3. Selects optimal action using epsilon-greedy Q-learning.
    4. Records the experience event.
    """
    if not req.context_vector:
        raise HTTPException(status_code=400, detail="context_vector cannot be empty.")

    # 1. Online Euclidean clustering
    state_id, is_new_state, cluster_dist = find_or_create_cluster(req.context_vector)

    # 2. Markov state transition & prediction
    predicted_next_state = record_markov_and_predict(req.agent_id, state_id)

    # 3. Action selection
    selected_action, is_exploration = choose_action(state_id, req.available_actions or [])

    # 4. Save experience
    event_id = db_insert_experience(
        agent_id=req.agent_id,
        modality=req.modality,
        value=req.value,
        state_id=state_id,
        predicted_next_state=predicted_next_state,
        action=selected_action
    )

    return {
        "event_id": event_id,
        "state_id": state_id,
        "predicted_next_state": predicted_next_state,
        "selected_action": selected_action,
        "is_new_state": is_new_state,
        "cluster_distance": round(cluster_dist, 4) if cluster_dist is not None else 0.0,
        "exploration": is_exploration
    }

@app.post("/api/feedback")
async def post_feedback(req: FeedbackRequest):
    """
    Accepts feedback for a recorded event_id and updates experience reward.
    """
    exp = db_query("SELECT id, state_id, action FROM experiences WHERE id = ?", (req.event_id,), one=True)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Event {req.event_id} not found.")

    db_execute(
        "UPDATE experiences SET actual_reward = ?, has_feedback = 1, consolidated = 0 WHERE id = ?",
        (req.actual_reward, req.event_id)
    )

    return {
        "status": "success",
        "event_id": req.event_id,
        "actual_reward": req.actual_reward,
        "message": "Feedback recorded. Queued for background Q-replay."
    }

@app.get("/api/state")
async def get_state():
    """
    Returns discovered clusters, Q-table rows, recent experience logs,
    and a 30-point chronological history of rolling average reward and discovered state count.
    """
    # 1. Clusters
    raw_clusters = db_query("SELECT state_id, centroid, vector_length, sample_count, updated_at FROM clusters ORDER BY state_id ASC") or []
    clusters = []
    for c in raw_clusters:
        try:
            centroid_parsed = json.loads(c["centroid"])
        except Exception:
            centroid_parsed = []
        clusters.append({
            "state_id": c["state_id"],
            "centroid": centroid_parsed,
            "vector_length": c["vector_length"],
            "sample_count": c["sample_count"],
            "updated_at": str(c["updated_at"])
        })

    # 2. Q-Table
    raw_q = db_query("SELECT state_id, action, q_value, visit_count, updated_at FROM q_table ORDER BY state_id ASC, q_value DESC") or []
    q_table = [{
        "state_id": r["state_id"],
        "action": r["action"],
        "q_value": round(float(r["q_value"]), 4),
        "visit_count": r["visit_count"],
        "updated_at": str(r["updated_at"])
    } for r in raw_q]

    # 3. Recent logs (last 50)
    raw_logs = db_query("""
        SELECT id, agent_id, modality, value, state_id, predicted_next_state, action,
               actual_reward, has_feedback, consolidated, created_at
        FROM experiences
        ORDER BY id DESC
        LIMIT 50
    """) or []

    recent_logs = [{
        "id": r["id"],
        "agent_id": r["agent_id"],
        "modality": r["modality"],
        "value": round(float(r["value"]), 3),
        "state_id": r["state_id"],
        "predicted_next_state": r["predicted_next_state"],
        "action": r["action"],
        "actual_reward": round(float(r["actual_reward"]), 3) if r["actual_reward"] is not None else None,
        "has_feedback": bool(r["has_feedback"]),
        "consolidated": bool(r["consolidated"]),
        "created_at": str(r["created_at"])
    } for r in raw_logs]

    # 4. Markov Transitions
    raw_transitions = db_query("SELECT from_state, to_state, transition_count FROM markov_transitions ORDER BY from_state ASC, transition_count DESC") or []
    markov_transitions = [{
        "from_state": r["from_state"],
        "to_state": r["to_state"],
        "transition_count": r["transition_count"]
    } for r in raw_transitions]

    # 5. Chronological 30-point History (Rolling Reward vs Discovered States)
    snapshots = db_query("SELECT rolling_reward, discovered_states_count, created_at FROM history_snapshots ORDER BY id DESC LIMIT 30") or []
    snapshots.reverse()  # chronological order

    # Current stats
    current_states_count = len(clusters)
    recent_rewards = [r["actual_reward"] for r in recent_logs if r["actual_reward"] is not None]
    current_rolling_reward = round(sum(recent_rewards) / len(recent_rewards), 4) if recent_rewards else 0.0

    history = []
    if snapshots:
        for s in snapshots:
            time_label = str(s["created_at"])[11:19] if len(str(s["created_at"])) >= 19 else str(s["created_at"])
            history.append({
                "timestamp": time_label or "now",
                "rolling_reward": round(float(s["rolling_reward"]), 4),
                "discovered_states": int(s["discovered_states_count"])
            })

    # Pad up to 30 points so chart always renders complete series
    if len(history) < 30:
        needed = 30 - len(history)
        now_ts = int(time.time())
        padded = []
        for i in range(needed):
            past_time = datetime.fromtimestamp(now_ts - (needed - i) * 5).strftime("%H:%M:%S")
            padded.append({
                "timestamp": past_time,
                "rolling_reward": current_rolling_reward,
                "discovered_states": current_states_count
            })
        history = padded + history

    # Additional summary metrics
    total_exp_row = db_query("SELECT COUNT(*) as count FROM experiences", one=True)
    total_experiences = total_exp_row["count"] if total_exp_row else 0

    return {
        "status": "online",
        "db_type": DB_TYPE,
        "db_name": DB_NAME if DB_TYPE == "sqlite" else "PostgreSQL (Remote)",
        "threshold": EUCLIDEAN_THRESHOLD,
        "epsilon": EPSILON,
        "metrics": {
            "total_experiences": total_experiences,
            "discovered_states_count": current_states_count,
            "q_table_rows": len(q_table),
            "rolling_reward": current_rolling_reward
        },
        "clusters": clusters,
        "q_table": q_table,
        "markov_transitions": markov_transitions,
        "recent_logs": recent_logs,
        "history": history
    }


# ==========================================
# Frontend Dashboard (Dark Mode + Chart.js)
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves the live dark-mode web dashboard with Chart.js dual-axis visualization,
    one-click interactive simulation buttons, and live-updating tables.
    """
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous Predictive Learning Node</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg-base: #090d16;
      --bg-card: #111827;
      --bg-card-subtle: #172033;
      --border-color: #1f293d;
      --border-highlight: #2d3b55;
      --text-main: #f3f4f6;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-purple: #a855f7;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      padding: 1.5rem;
      min-height: 100vh;
      line-height: 1.5;
    }
    /* Layout */
    .container {
      max-width: 1480px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    /* Header */
    header {
      background: linear-gradient(135deg, #111827 0%, #172033 100%);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem 1.75rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .brand-title {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .pulse-dot {
      width: 12px;
      height: 12px;
      background-color: var(--accent-emerald);
      border-radius: 50%;
      box-shadow: 0 0 12px var(--accent-emerald);
      animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }

    /* Top Stats Grid */
    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 1.1rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, var(--border-highlight), transparent);
    }
    .stat-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    .stat-value {
      font-size: 1.65rem;
      font-weight: 700;
      color: #fff;
    }
    .stat-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Simulation Control Bar */
    .controls-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .controls-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
    }
    .controls-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .button-group {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
    }
    button {
      cursor: pointer;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 0.55rem 0.95rem;
      font-size: 0.825rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease-in-out;
      background: var(--bg-card-subtle);
      color: var(--text-main);
      border-color: var(--border-color);
    }
    button:hover {
      background: #1e293b;
      border-color: var(--border-highlight);
      transform: translateY(-1px);
    }
    button:active {
      transform: translateY(0);
    }
    .btn-primary {
      background: #0284c7;
      border-color: #0369a1;
      color: #fff;
    }
    .btn-primary:hover {
      background: #0369a1;
    }
    .btn-emerald {
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.4);
      color: #34d399;
    }
    .btn-emerald:hover {
      background: rgba(16, 185, 129, 0.35);
    }
    .btn-purple {
      background: rgba(168, 85, 247, 0.2);
      border-color: rgba(168, 85, 247, 0.4);
      color: #c084fc;
    }
    .btn-purple:hover {
      background: rgba(168, 85, 247, 0.35);
    }
    .btn-rose {
      background: rgba(244, 63, 94, 0.2);
      border-color: rgba(244, 63, 94, 0.4);
      color: #fb7185;
    }
    .btn-rose:hover {
      background: rgba(244, 63, 94, 0.35);
    }
    .btn-active {
      background: #10b981;
      border-color: #059669;
      color: #fff;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }

    /* Chart Card */
    .chart-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem;
      position: relative;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .chart-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .chart-container {
      position: relative;
      height: 320px;
      width: 100%;
    }

    /* Grid for Tables */
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }
    @media (max-width: 1080px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    .table-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      overflow: hidden;
    }
    .table-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .table-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .table-wrap {
      overflow-x: auto;
      max-height: 320px;
      border: 1px solid var(--border-color);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.8rem;
    }
    th {
      background: #172033;
      padding: 0.6rem 0.8rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 0.04em;
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--border-highlight);
    }
    td {
      padding: 0.55rem 0.8rem;
      border-bottom: 1px solid #1a2333;
      color: var(--text-main);
      white-space: nowrap;
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }
    .font-mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      background: #1e293b;
      border: 1px solid var(--border-highlight);
      border-radius: 8px;
      padding: 0.75rem 1.25rem;
      font-size: 0.85rem;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px);
      transition: all 0.2s ease-in-out;
      z-index: 1000;
    }
    #toast.show {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }
  </style>
</head>
<body>

<div class="container">

  <!-- Header -->
  <header>
    <div class="brand-title">
      <div class="pulse-dot"></div>
      <div>
        <h1>Autonomous Predictive Learning Node</h1>
        <p style="font-size: 0.78rem; color: var(--text-muted);">
          Online Euclidean Clustering (thresh: <span class="badge badge-cyan" id="node-thresh">0.45</span>) &bull;
          Markov Next-State Prediction &bull;
          &epsilon;-Greedy Q-Learning
        </p>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
      <span class="badge badge-emerald" id="db-badge">DB: Detecting...</span>
      <span class="badge badge-purple" id="worker-badge">Replay Worker: 5s Cycle</span>
      <span style="font-size: 0.75rem; color: var(--text-muted);" id="refresh-timer">Auto-sync: 2s</span>
      <button onclick="fetchState(true)" title="Force Refresh">&#x21bb; Refresh</button>
    </div>
  </header>

  <!-- Metrics Cards -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">Discovered States</div>
      <div class="stat-value" id="stat-states" style="color: var(--accent-purple);">0</div>
      <div class="stat-sub">Self-partitioned Euclidean clusters</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Rolling Average Reward</div>
      <div class="stat-value" id="stat-reward" style="color: var(--accent-cyan);">0.000</div>
      <div class="stat-sub">Last 30 consolidated experiences</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Signals Ingested</div>
      <div class="stat-value" id="stat-signals" style="color: var(--accent-emerald);">0</div>
      <div class="stat-sub">Experiences logged &amp; buffered</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Q-Policy Matrix Entries</div>
      <div class="stat-value" id="stat-qentries" style="color: var(--accent-amber);">0</div>
      <div class="stat-sub">State &times; Action learned values</div>
    </div>
  </div>

  <!-- Interactive Simulation Bar -->
  <div class="controls-panel">
    <div class="controls-header">
      <div class="controls-title">
        <span>⚡ Interactive Signal Generator</span>
        <span style="font-size: 0.75rem; font-weight: normal; color: var(--text-muted);">(Test the autonomous node with live one-click signals)</span>
      </div>
      <div class="button-group">
        <button class="btn-emerald" onclick="simulateSignal('anomaly')">⚡ Simulate Anomaly [0.92, 0.88, 0.95]</button>
        <button class="btn-primary" onclick="simulateSignal('baseline')">🔄 Simulate Baseline [0.12, 0.15, 0.11]</button>
        <button class="btn-purple" onclick="simulateSignal('load')">📈 Simulate Surge [0.65, 0.70, 0.62]</button>
        <button onclick="simulateSignal('random')">🎲 Random Vector</button>
        <button id="auto-btn" onclick="toggleAutoStream()">⏱️ Auto-Stream: OFF</button>
        <button onclick="toggleCustomPanel()">🛠️ Custom Vector</button>
      </div>
    </div>
    <div id="custom-panel" style="display: none; background: #0b1120; border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; margin-top: 0.25rem;">
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;">
        <div>
          <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Agent ID</label>
          <input id="custom-agent" type="text" value="agent-custom" style="width: 100%; background: #172033; border: 1px solid var(--border-color); color: #fff; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem;">
        </div>
        <div>
          <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Modality</label>
          <input id="custom-modality" type="text" value="high_dim_telemetry" style="width: 100%; background: #172033; border: 1px solid var(--border-color); color: #fff; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem;">
        </div>
        <div>
          <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Scalar Value</label>
          <input id="custom-val" type="number" step="0.1" value="45.2" style="width: 100%; background: #172033; border: 1px solid var(--border-color); color: #fff; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem;">
        </div>
        <div>
          <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Context Vector (comma-separated, any length)</label>
          <input id="custom-vec" type="text" value="0.22, 0.45, 0.68, 0.91" style="width: 100%; background: #172033; border: 1px solid var(--border-color); color: #fff; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-family: monospace;">
        </div>
        <div>
          <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Available Actions (comma-separated)</label>
          <input id="custom-actions" type="text" value="SCALE_UP, THROTTLE, MAINTAIN, REDIRECT" style="width: 100%; background: #172033; border: 1px solid var(--border-color); color: #fff; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem;">
        </div>
      </div>
      <div style="margin-top: 0.75rem; display: flex; justify-content: flex-end;">
        <button class="btn-primary" onclick="submitCustomSignal()">🚀 Transmit Vector to Node</button>
      </div>
    </div>
  </div>

  <!-- Chart.js Dual Axis Line Chart -->
  <div class="chart-card">
    <div class="chart-header">
      <div class="chart-title">
        <span>📈 Live Dual-Axis Dynamics</span>
        <span style="font-size: 0.75rem; color: var(--text-muted);">(Rolling Reward on Left Axis &bull; Discovered State Count on Right Axis)</span>
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted);">Chronological 30-Point History</div>
    </div>
    <div class="chart-container">
      <canvas id="dualAxisChart"></canvas>
    </div>
  </div>

  <!-- Tables Section 1: Clusters & Q-Table -->
  <div class="grid-2">
    <!-- Discovered Clusters Table -->
    <div class="table-card">
      <div class="table-header">
        <div class="table-title">
          <span>🧠 Discovered Clusters (State Space)</span>
          <span class="badge badge-purple" id="badge-cluster-count">0 States</span>
        </div>
        <span style="font-size: 0.72rem; color: var(--text-muted);">Online Centroid Update (d &le; 0.45)</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>State ID</th>
              <th>Dimension</th>
              <th>Sample Count</th>
              <th>Centroid Vector</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody id="clusters-tbody">
            <tr><td colspan="5" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">Awaiting first signal to discover clusters...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Q-Table (Learned Policy) -->
    <div class="table-card">
      <div class="table-header">
        <div class="table-title">
          <span>🎯 Q-Table (Learned Action Policy)</span>
          <span class="badge badge-amber" id="badge-q-count">0 Actions</span>
        </div>
        <span style="font-size: 0.72rem; color: var(--text-muted);">&epsilon;=0.15 &bull; &alpha;=0.1 &bull; &gamma;=0.9</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>State</th>
              <th>Action</th>
              <th>Q-Value</th>
              <th>Replays</th>
              <th>Policy</th>
            </tr>
          </thead>
          <tbody id="qtable-tbody">
            <tr><td colspan="5" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">Q-table updates during 5-second background replay...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Tables Section 2: Recent Logs with Interactive Feedback -->
  <div class="table-card">
    <div class="table-header">
      <div class="table-title">
        <span>📜 Recent Experience Stream &amp; Live Reward Controls</span>
        <span class="badge badge-cyan" id="badge-logs-count">0 Events</span>
      </div>
      <span style="font-size: 0.72rem; color: var(--text-muted);">Directly award +1.0 or -1.0 feedback to train the Q-table in real-time</span>
    </div>
    <div class="table-wrap" style="max-height: 380px;">
      <table>
        <thead>
          <tr>
            <th>Event ID</th>
            <th>Agent &amp; Modality</th>
            <th>Value</th>
            <th>State</th>
            <th>Markov Next Pred</th>
            <th>Action</th>
            <th>Reward</th>
            <th>Feedback Action</th>
            <th>Consolidation</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody id="logs-tbody">
          <tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">No experience logs recorded yet. Click one of the simulation buttons above!</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Markov Transitions (State Transitions Matrix) -->
  <div class="table-card">
    <div class="table-header">
      <div class="table-title">
        <span>🔄 Markov State Transitions</span>
        <span class="badge badge-emerald" id="badge-markov-count">0 Transitions</span>
      </div>
      <span style="font-size: 0.72rem; color: var(--text-muted);">Probabilistic next-state sequence learning</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>From State</th>
            <th>To State</th>
            <th>Transition Frequency</th>
            <th>Confidence / Transition Path</th>
          </tr>
        </thead>
        <tbody id="markov-tbody">
          <tr><td colspan="4" style="text-align:center; color: var(--text-muted); padding: 1.25rem;">Awaiting state transitions...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<!-- Toast Container -->
<div id="toast">
  <span id="toast-icon">ℹ️</span>
  <span id="toast-msg">Notification</span>
</div>

<script>
  let autoStreamInterval = null;
  let chartInstance = null;

  // Initialize Chart.js Dual Axis Chart
  function initChart() {
    const ctx = document.getElementById('dualAxisChart').getContext('2d');
    chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Rolling Average Reward',
            data: [],
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.08)',
            borderWidth: 2.5,
            pointRadius: 3,
            pointHoverRadius: 6,
            pointBackgroundColor: '#06b6d4',
            tension: 0.3,
            fill: true,
            yAxisID: 'yReward',
          },
          {
            label: 'Discovered States Count',
            data: [],
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.05)',
            borderWidth: 2,
            borderDash: [4, 4],
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: '#a855f7',
            stepped: false,
            tension: 0.2,
            yAxisID: 'yStates',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            labels: {
              color: '#94a3b8',
              font: { size: 11, weight: '500' }
            }
          },
          tooltip: {
            backgroundColor: '#111827',
            titleColor: '#fff',
            bodyColor: '#cbd5e1',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            grid: { color: '#1f2937' },
            ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 0 }
          },
          yReward: {
            type: 'linear',
            display: true,
            position: 'left',
            grid: { color: '#1f2937' },
            ticks: { color: '#06b6d4', font: { size: 10 } },
            title: {
              display: true,
              text: 'Rolling Avg Reward',
              color: '#06b6d4',
              font: { size: 11 }
            }
          },
          yStates: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#a855f7', font: { size: 10 }, stepSize: 1 },
            title: {
              display: true,
              text: 'Discovered States Count',
              color: '#a855f7',
              font: { size: 11 }
            }
          }
        }
      }
    });
  }

  // Toast helper
  function showToast(msg, icon = '✓') {
    const t = document.getElementById('toast');
    document.getElementById('toast-msg').innerText = msg;
    document.getElementById('toast-icon').innerText = icon;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2500);
  }

  // Fetch state from GET /api/state
  async function fetchState(manual = false) {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();

      // Update badges
      document.getElementById('db-badge').innerText = 'DB: ' + (data.db_type === 'postgres' ? 'PostgreSQL' : 'SQLite (mind.db)');
      document.getElementById('node-thresh').innerText = data.threshold;

      // Update Top Stats
      document.getElementById('stat-states').innerText = data.metrics.discovered_states_count;
      document.getElementById('stat-reward').innerText = data.metrics.rolling_reward.toFixed(3);
      document.getElementById('stat-signals').innerText = data.metrics.total_experiences;
      document.getElementById('stat-qentries').innerText = data.metrics.q_table_rows;

      // Update Chart
      if (chartInstance && data.history) {
        chartInstance.data.labels = data.history.map(h => h.timestamp);
        chartInstance.data.datasets[0].data = data.history.map(h => h.rolling_reward);
        chartInstance.data.datasets[1].data = data.history.map(h => h.discovered_states);
        chartInstance.update('none');
      }

      // Render Clusters Table
      renderClusters(data.clusters);
      document.getElementById('badge-cluster-count').innerText = data.clusters.length + ' States';

      // Render Q-Table
      renderQTable(data.q_table);
      document.getElementById('badge-q-count').innerText = data.q_table.length + ' Pairs';

      // Render Logs Table
      renderLogs(data.recent_logs);
      document.getElementById('badge-logs-count').innerText = data.recent_logs.length + ' Recent';

      // Render Markov Table
      renderMarkov(data.markov_transitions);
      document.getElementById('badge-markov-count').innerText = data.markov_transitions.length + ' Transitions';

      if (manual) showToast('State refreshed from node', '✓');
    } catch (e) {
      console.error('fetchState error:', e);
    }
  }

  function renderClusters(clusters) {
    const tbody = document.getElementById('clusters-tbody');
    if (!clusters || clusters.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">Awaiting first signal to discover clusters...</td></tr>';
      return;
    }
    tbody.innerHTML = clusters.map(c => {
      const vectorPreview = c.centroid.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ');
      return `
        <tr>
          <td><span class="badge badge-purple">${c.state_id}</span></td>
          <td class="font-mono">${c.vector_length}D</td>
          <td><strong style="color:#fff;">${c.sample_count}</strong> samples</td>
          <td class="font-mono" style="color:#94a3b8; max-width: 250px; overflow: hidden; text-overflow: ellipsis;" title="[${vectorPreview}]">[${vectorPreview}]</td>
          <td style="color:#64748b; font-size:0.75rem;">${c.updated_at}</td>
        </tr>
      `;
    }).join('');
  }

  function renderQTable(qtable) {
    const tbody = document.getElementById('qtable-tbody');
    if (!qtable || qtable.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">Q-table updates during 5-second background replay...</td></tr>';
      return;
    }

    // Group by state to find max action
    const maxMap = {};
    qtable.forEach(r => {
      if (!(r.state_id in maxMap) || r.q_value > maxMap[r.state_id]) {
        maxMap[r.state_id] = r.q_value;
      }
    });

    tbody.innerHTML = qtable.map(q => {
      const isBest = (q.q_value === maxMap[q.state_id] && q.q_value !== 0);
      let qColor = '#94a3b8';
      if (q.q_value > 0.001) qColor = '#10b981';
      else if (q.q_value < -0.001) qColor = '#f43f5e';

      return `
        <tr>
          <td><span class="badge badge-purple">${q.state_id}</span></td>
          <td class="font-mono"><strong style="color:#f8fafc;">${q.action}</strong></td>
          <td class="font-mono" style="color:${qColor}; font-weight:700;">${q.q_value > 0 ? '+' : ''}${q.q_value.toFixed(4)}</td>
          <td>${q.visit_count}</td>
          <td>${isBest ? '<span class="badge badge-emerald">&#9733; Optimal</span>' : '<span style="color:#64748b;">Suboptimal</span>'}</td>
        </tr>
      `;
    }).join('');
  }

  function renderLogs(logs) {
    const tbody = document.getElementById('logs-tbody');
    if (!logs || logs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">No experience logs recorded yet. Click one of the simulation buttons above!</td></tr>';
      return;
    }
    tbody.innerHTML = logs.map(l => {
      let rewardBadge = '<span style="color:#64748b;">Unrated</span>';
      if (l.actual_reward !== null) {
        if (l.actual_reward > 0) rewardBadge = `<span class="badge badge-emerald">+${l.actual_reward}</span>`;
        else if (l.actual_reward < 0) rewardBadge = `<span class="badge badge-rose">${l.actual_reward}</span>`;
        else rewardBadge = `<span class="badge badge-cyan">0.0</span>`;
      }

      const consolBadge = l.consolidated
        ? '<span class="badge badge-emerald">&#10003; Replayed</span>'
        : '<span class="badge badge-amber">&#8987; Pending (5s)</span>';

      return `
        <tr>
          <td class="font-mono" style="color:#94a3b8;">#${l.id}</td>
          <td><strong>${l.agent_id}</strong> &bull; <span style="color:#94a3b8;">${l.modality}</span></td>
          <td class="font-mono">${l.value}</td>
          <td><span class="badge badge-purple">${l.state_id}</span></td>
          <td><span class="badge badge-cyan">&rarr; ${l.predicted_next_state}</span></td>
          <td><span class="badge badge-amber">${l.action}</span></td>
          <td>${rewardBadge}</td>
          <td>
            <div style="display:flex; gap:0.25rem;">
              <button onclick="sendFeedback(${l.id}, 1.0)" style="padding:0.2rem 0.45rem; font-size:0.72rem;" class="btn-emerald" title="Send Positive Feedback">+1.0</button>
              <button onclick="sendFeedback(${l.id}, -1.0)" style="padding:0.2rem 0.45rem; font-size:0.72rem;" class="btn-rose" title="Send Negative Feedback">-1.0</button>
            </div>
          </td>
          <td>${consolBadge}</td>
          <td style="color:#64748b; font-size:0.72rem;">${l.created_at}</td>
        </tr>
      `;
    }).join('');
  }

  function renderMarkov(transitions) {
    const tbody = document.getElementById('markov-tbody');
    if (!transitions || transitions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--text-muted); padding: 1.25rem;">Awaiting state transitions...</td></tr>';
      return;
    }
    // Calculate total transitions per from_state for percentage
    const totals = {};
    transitions.forEach(t => {
      totals[t.from_state] = (totals[t.from_state] || 0) + t.transition_count;
    });

    tbody.innerHTML = transitions.map(t => {
      const pct = Math.round((t.transition_count / totals[t.from_state]) * 100);
      return `
        <tr>
          <td><span class="badge badge-purple">${t.from_state}</span></td>
          <td><span class="badge badge-cyan">${t.to_state}</span></td>
          <td class="font-mono"><strong style="color:#fff;">${t.transition_count}</strong> times</td>
          <td>
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <div style="background:#1f2937; border-radius:4px; height:8px; width:80px; overflow:hidden;">
                <div style="background:#06b6d4; height:100%; width:${pct}%;"></div>
              </div>
              <span class="font-mono" style="font-size:0.75rem; color:#94a3b8;">${pct}%</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  // Interactive Signal Dispatcher
  async function simulateSignal(type) {
    let payload = {};
    if (type === 'anomaly') {
      payload = {
        agent_id: "edge-sensor-alpha",
        modality: "thermal_anomaly",
        value: 94.2,
        context_vector: [0.92, 0.88, 0.95],
        available_actions: ["THROTTLE", "FAILOVER", "ALERT", "MAINTAIN"]
      };
    } else if (type === 'baseline') {
      payload = {
        agent_id: "edge-sensor-beta",
        modality: "nominal_flow",
        value: 12.4,
        context_vector: [0.12, 0.15, 0.11],
        available_actions: ["MAINTAIN", "OPTIMIZE", "SCALE_DOWN"]
      };
    } else if (type === 'load') {
      payload = {
        agent_id: "gateway-node-01",
        modality: "request_surge",
        value: 78.5,
        context_vector: [0.65, 0.70, 0.62],
        available_actions: ["SCALE_UP", "SHED_LOAD", "MAINTAIN"]
      };
    } else {
      // random vector
      const v = [
        Math.round(Math.random() * 100) / 100,
        Math.round(Math.random() * 100) / 100,
        Math.round(Math.random() * 100) / 100
      ];
      payload = {
        agent_id: "agent-" + Math.floor(Math.random() * 5 + 1),
        modality: "stochastic_probe",
        value: Math.round(Math.random() * 1000) / 10,
        context_vector: v,
        available_actions: ["ADAPT", "EXPLORE", "STABILIZE"]
      };
    }

    try {
      const res = await fetch('/api/signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      showToast(`Signal &rarr; ${data.state_id} | Action: ${data.selected_action} ${data.is_new_state ? '(Discovered New State!)' : ''}`, '⚡');
      await fetchState();
    } catch (err) {
      showToast('Error sending signal: ' + err, '⚠️');
    }
  }

  // Interactive Feedback Dispatcher
  async function sendFeedback(eventId, reward) {
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event_id: eventId, actual_reward: reward })
      });
      const data = await res.json();
      showToast(`Event #${eventId} reward set to ${reward}. Replay scheduled!`, '🎯');
      await fetchState();
    } catch (err) {
      showToast('Error recording feedback: ' + err, '⚠️');
    }
  }

  // Toggle custom signal tester panel
  function toggleCustomPanel() {
    const p = document.getElementById('custom-panel');
    p.style.display = (p.style.display === 'none' || !p.style.display) ? 'block' : 'none';
  }

  // Submit custom signal from inputs
  async function submitCustomSignal() {
    const agent_id = document.getElementById('custom-agent').value.trim() || 'agent-custom';
    const modality = document.getElementById('custom-modality').value.trim() || 'telemetry';
    const val = parseFloat(document.getElementById('custom-val').value) || 0.0;
    const rawVec = document.getElementById('custom-vec').value.trim();
    const rawActions = document.getElementById('custom-actions').value.trim();

    const vector = rawVec.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (vector.length === 0) {
      showToast('Please enter at least one valid float in the context vector', '⚠️');
      return;
    }
    const actions = rawActions.split(',').map(s => s.trim()).filter(s => s.length > 0);

    const payload = {
      agent_id: agent_id,
      modality: modality,
      value: val,
      context_vector: vector,
      available_actions: actions.length > 0 ? actions : ["MAINTAIN", "SCALE_UP", "THROTTLE"]
    };

    try {
      const res = await fetch('/api/signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      showToast(`Signal (${vector.length}D) &rarr; ${data.state_id} | ${data.selected_action} (Dist: ${data.cluster_distance})`, '🚀');
      await fetchState();
    } catch (err) {
      showToast('Error transmitting custom signal: ' + err, '⚠️');
    }
  }

  // Auto-stream toggle
  function toggleAutoStream() {
    const btn = document.getElementById('auto-btn');
    if (autoStreamInterval) {
      clearInterval(autoStreamInterval);
      autoStreamInterval = null;
      btn.innerText = '⏱️ Auto-Stream: OFF';
      btn.classList.remove('btn-active');
      showToast('Auto-stream stopped', '⏹');
    } else {
      btn.innerText = '⏱️ Auto-Stream: ACTIVE';
      btn.classList.add('btn-active');
      showToast('Auto-stream running (signal every 2.5s)', '▶');
      const types = ['anomaly', 'baseline', 'load', 'random'];
      let i = 0;
      autoStreamInterval = setInterval(async () => {
        await simulateSignal(types[i % types.length]);
        i++;
      }, 2500);
    }
  }

  // Auto-refresh every 2 seconds
  window.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchState();
    setInterval(fetchState, 2000);
  });
</script>

</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)


# ==========================================
# Run script directly if invoked
# ==========================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
