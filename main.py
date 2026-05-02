"""
main.py — Vera Message Engine
FastAPI server implementing the exact magicpin judge harness API contract.

Endpoints (per challenge-testing-brief.md §2):
  GET  /v1/healthz  — liveness + context counts
  GET  /v1/metadata — bot identity
  POST /v1/context  — idempotent context push (200 or 409)
  POST /v1/tick     — proactive message generation
  POST /v1/reply    — conversation reply handling
"""

import os, time, json, logging
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

import database
from security import check_prompt_injection, injection_response
from llm_pipeline import compose_tick_action, compose_reply

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vera.main")

START_TIME = time.time()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Vera Message Engine starting...")
    database.get_db()
    database.wipe_all()
    logger.info("Database ready and wiped ✓")
    yield
    database.close_db()
    logger.info("Shutdown complete ✓")


app = FastAPI(title="Vera Message Engine", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logger_traffic = logging.getLogger("vera.traffic")

@app.middleware("http")
async def log_traffic(request: Request, call_next):
    start_time = time.time()
    
    # 1. Log the incoming hit
    logger_traffic.info(f"[JUDGE IN] {request.method} {request.url.path}")
    
    # 2. Let the app process the request
    response = await call_next(request)
    
    # 3. Log the result and speed
    process_time = time.time() - start_time
    logger_traffic.info(f"[JUDGE OUT] {request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time:.3f}s")
    
    return response


# ─── Pydantic Models (exact judge contract) ──────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Any = Field(default_factory=dict)
    delivered_at: Optional[str] = None

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int


# ─── GET /v1/healthz ──────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = database.count_contexts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# ─── GET /v1/metadata ────────────────────────────────────────────────────────

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Engine",
        "team_members": ["Madhav Kapila"],
        "model": "cerebras/llama3.1-8b + groq/llama-3.3-70b-versatile",
        "approach": "Tri-Model Pipeline: Cerebras diagnostician extracts signal, Groq copywriter composes. Category voice routing. Prompt Guard security shield.",
        "contact_email": "madhav@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ─── POST /v1/context ────────────────────────────────────────────────────────

@app.post("/v1/context")
async def push_context(body: ContextBody):
    logger.info(f"[PAYLOAD] /v1/context triggered with: {body.model_dump_json()}")
    scope = body.scope.lower().strip()
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })

    result = database.upsert_context(
        scope=scope, context_id=body.context_id,
        version=body.version, payload=body.payload,
        delivered_at=body.delivered_at,
    )

    status_code = result.pop("status_code", 200)
    if status_code == 409:
        return JSONResponse(status_code=409, content=result)
    return result


# ─── POST /v1/tick ────────────────────────────────────────────────────────────

@app.post("/v1/tick")
async def tick(body: TickBody):
    logger.info(f"[PAYLOAD] /v1/tick triggered with: {body.model_dump_json()}")
    actions = []

    for trig_id in body.available_triggers:
        try:
            # Fetch trigger context
            trig_ctx = database.get_context("trigger", trig_id)
            if not trig_ctx:
                logger.warning("Trigger %s not found in DB, skipping", trig_id)
                continue
            trigger = trig_ctx["payload"]

            # Fetch merchant context
            merchant_id = trigger.get("merchant_id")
            if not merchant_id:
                continue
            merch_ctx = database.get_context("merchant", merchant_id)
            if not merch_ctx:
                logger.warning("Merchant %s not found for trigger %s", merchant_id, trig_id)
                continue
            merchant = merch_ctx["payload"]

            # Fetch category context
            cat_slug = merchant.get("category_slug", "")
            cat_ctx = database.get_context("category", cat_slug)
            category = cat_ctx["payload"] if cat_ctx else {"slug": cat_slug}

            # Fetch customer context if customer-scoped
            customer = None
            customer_id = trigger.get("customer_id")
            if customer_id:
                cust_ctx = database.get_context("customer", customer_id)
                if cust_ctx:
                    customer = cust_ctx["payload"]

            # Run the compose pipeline
            action = compose_tick_action(category, merchant, trigger, customer)

            if action and action.get("body"):
                actions.append(action)
                # Log the bot's outbound message
                database.append_turn(
                    action["conversation_id"], 1, "vera", action["body"]
                )

        except Exception as e:
            logger.error("Error processing trigger %s: %s", trig_id, e, exc_info=True)
            continue

    return {"actions": actions}


# ─── POST /v1/reply ──────────────────────────────────────────────────────────

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    logger.info(f"[PAYLOAD] /v1/reply triggered with: {body.model_dump_json()}")
    # Security Shield: check for prompt injection FIRST
    is_safe = check_prompt_injection(body.message)
    if not is_safe:
        logger.warning("Injection blocked: conv=%s", body.conversation_id)
        database.append_turn(body.conversation_id, body.turn_number, body.from_role, "[BLOCKED]")
        return injection_response()

    # Log the inbound message
    database.append_turn(
        body.conversation_id, body.turn_number, body.from_role, body.message
    )

    # Fetch merchant context
    merchant = {}
    category = {}
    if body.merchant_id:
        merch_ctx = database.get_context("merchant", body.merchant_id)
        if merch_ctx:
            merchant = merch_ctx["payload"]
            cat_slug = merchant.get("category_slug", "")
            cat_ctx = database.get_context("category", cat_slug)
            category = cat_ctx["payload"] if cat_ctx else {"slug": cat_slug}

    # Fetch conversation history
    conv_history = database.get_conversation(body.conversation_id)

    # Fetch customer context if provided
    customer = None
    if body.customer_id:
        cust_ctx = database.get_context("customer", body.customer_id)
        if cust_ctx:
            customer = cust_ctx["payload"]

    # Run compose pipeline in reply mode
    try:
        result = compose_reply(
            merchant=merchant, category=category, message=body.message,
            conversation_history=conv_history, customer=customer,
        )
    except Exception as e:
        logger.error("Reply compose error: %s", e, exc_info=True)
        result = {"action": "send", "body": "Got it — let me look into that for you.",
                  "cta": "open_ended", "rationale": "Fallback due to processing error"}

    # Log bot's response
    if result.get("body"):
        database.append_turn(
            body.conversation_id, body.turn_number + 1, "vera", result["body"]
        )

    return result


# ─── POST /v1/teardown (optional per spec §11) ──────────────────────────────

@app.post("/v1/teardown")
async def teardown():
    database.wipe_all()
    return {"status": "wiped"}


# ─── Payload Size Guard ─────────────────────────────────────────────────────

@app.middleware("http")
async def payload_guard(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > 512_000:
        return JSONResponse(status_code=413, content={"error": "Payload too large"})
    return await call_next(request)


# ─── Global Error Handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", workers=1)
