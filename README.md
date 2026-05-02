# Vera Message Engine

**Stateful, deterministic message engine for merchant engagement — magicpin AI Challenge 2026**

## Architecture

```
Request → Payload Limiter → Security Shield (Prompt Guard) → State Manager (SQLite) → Tri-Model Pipeline → Response
```

### Tri-Model Pipeline

| Step | Provider | Model | Role |
|------|----------|-------|------|
| 1. Diagnostician | Cerebras | llama3.1-8b | Extracts the ONE critical signal from merchant state |
| 2. Copywriter | Groq | llama-3.3-70b-versatile | Crafts high-compulsion message with specific CTA |
| Shield | Groq | llama-prompt-guard-2-86m | Blocks prompt injections before DB/LLM access |

### Category Routing (Pillar 5)

Dynamic system prompts tuned to merchant vertical:
- **Dentists/Pharmacies**: Clinical, utility-first, compliance-aware
- **Salons**: Visual, timely, aesthetic-focused
- **Restaurants**: Urgent, occasion-driven, locally grounded
- **Gyms**: Motivational, seasonal-dip reframing

## Setup

```bash
# 1. Clone and enter the project
cd VeraAgent

# 2. Copy and fill environment variables
cp .env.sample .env
# Edit .env with your Cerebras and Groq API keys

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python main.py
```

## Docker

```bash
docker build -t vera-engine .
docker run -p 8000:8000 --env-file .env vera-engine
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/healthz` | Health check (200 OK) |
| GET | `/v1/metadata` | Bot identity and capabilities |
| POST | `/v1/context` | Idempotent merchant context ingestion |
| POST | `/v1/tick` | Time simulation + proactive message generation |
| POST | `/v1/reply` | Reply handling + contextual response generation |

## Design Decisions

- **SQLite with WAL mode**: Zero-latency embedded state that survives container restarts
- **Temperature = 0.0**: All LLM calls are fully deterministic
- **Fail-open security**: If Prompt Guard is unreachable, requests are allowed through (logged)
- **Version-gated upserts**: Context updates are idempotent — same or lower version is a no-op
- **Fallback pipeline**: If any LLM is unavailable, the system uses heuristic signal extraction and template-based messages grounded in real merchant data

## Tradeoffs

1. **Single worker**: SQLite requires single-writer access. Traded concurrency for data integrity.
2. **Synchronous LLM calls in executor**: Cerebras/Groq SDKs are synchronous; wrapped in `run_in_executor` to avoid blocking the event loop.
3. **Prompt Guard fail-open**: Chose availability over strict security — a blocked legitimate request costs more than a logged suspicious one.
