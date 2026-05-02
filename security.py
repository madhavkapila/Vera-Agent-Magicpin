"""
security.py — Vera Message Engine
Prompt Guard middleware using Groq's meta-llama/llama-prompt-guard-2-86m.

Pillar 4: Security Shield
- Every inbound /v1/reply message passes through Prompt Guard BEFORE
  touching the DB or invoking any main LLM.
- On injection detection → return {"action": "end", "rationale": "Security violation detected."}
- Fail-open on API errors (logged) to avoid blocking legitimate requests.
"""

import os
import json
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("vera.security")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GUARD_MODEL = os.getenv("GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")


def check_prompt_injection(text: str) -> bool:
    """
    Run inbound text through Groq Prompt Guard.
    Returns True if the text is SAFE, False if injection detected.
    """
    if not text or not text.strip():
        return True  # Empty is safe

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — prompt guard DISABLED (fail-open)")
        return True

    try:
        resp = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GUARD_MODEL,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.0,
                "max_tokens": 32,
            },
            timeout=8,
        )

        if resp.status_code != 200:
            logger.error("Prompt Guard returned %d: %s", resp.status_code, resp.text[:300])
            return True  # Fail open

        data = resp.json()
        guard_output = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )

        # Check for injection indicators
        injection_words = ["unsafe", "injection", "jailbreak", "malicious", "attack", "yes"]
        safe_words = ["safe", "benign", "clean", "no injection", "legitimate", "no"]

        for w in safe_words:
            if w in guard_output:
                return True
        for w in injection_words:
            if w in guard_output:
                logger.warning("INJECTION DETECTED: %s → guard said: %s", text[:100], guard_output)
                return False

        return True  # Ambiguous → fail open

    except requests.Timeout:
        logger.warning("Prompt Guard timed out — fail-open")
        return True
    except Exception as e:
        logger.error("Prompt Guard error: %s", str(e))
        return True


def injection_response() -> Dict[str, Any]:
    """Return the security violation response per judge contract."""
    return {
        "action": "end",
        "rationale": "Security violation detected.",
    }
