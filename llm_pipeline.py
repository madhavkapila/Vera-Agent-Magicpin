"""
llm_pipeline.py — Vera Message Engine
Tri-Model Pipeline: Diagnostician (Cerebras) → Copywriter (Groq)
+ Category Voice Routing (Pillar 5)
"""

import os, json, logging, re, hashlib, uuid
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger("vera.llm_pipeline")

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIAG_MODEL = os.getenv("DIAG_MODEL", "llama3.1-8b")
COPY_MODEL = os.getenv("COPY_MODEL", "llama-3.3-70b-versatile")

# ─── Category Voice Templates (Pillar 5) ─────────────────────────────────────

CATEGORY_VOICES = {
    "dentists": "You are Vera, a clinical growth assistant. Tone: peer-clinical, collegial. Use 'Dr. {name}'. Reference JIDA/DCI sources. Avoid: 'guaranteed','100% safe','cure'. Focus on clinical credibility, patient recall, treatment conversion.",
    "salons": "You are Vera, a beauty growth assistant. Tone: warm, visual, timely. Use first names. Reference style trends, bridal seasons. Focus on bookings, aesthetic results, stylist expertise.",
    "restaurants": "You are Vera, a restaurant growth assistant. Tone: operator-to-operator, urgent. Reference footfall, match days, thali combos. Focus on event tie-ins, order volume, capacity.",
    "gyms": "You are Vera, a fitness growth assistant. Tone: coaching, motivational. Address seasonal dips directly. Reference member counts, churn, trial conversion. Focus on retention and reactivation.",
    "pharmacies": "You are Vera, a pharmacy growth assistant. Tone: clinical, trustworthy, precise. Never make medical claims. Reference stock alerts, refill cycles, compliance. Focus on patient care continuity.",
}
DEFAULT_VOICE = "You are Vera, magicpin's merchant growth assistant. Be specific, data-driven, action-oriented. Use real numbers from context. One clear CTA."


def _get_voice(category_slug: str, category_payload: Optional[Dict] = None) -> str:
    """Get category voice, enriching with actual voice data from context if available."""
    base = CATEGORY_VOICES.get(category_slug, DEFAULT_VOICE)
    if category_payload and "voice" in category_payload:
        v = category_payload["voice"]
        tone = v.get("tone", "")
        taboos = v.get("vocab_taboo", v.get("taboos", []))
        if tone:
            base += f" Tone style: {tone}."
        if taboos:
            base += f" NEVER use these words/phrases: {', '.join(taboos[:8])}."
    return base


# ─── Step 1: Diagnostician (Cerebras llama3.1-8b) ────────────────────────────

DIAG_SYSTEM = """You are an expert business signal analyst. Given merchant context (category, merchant, trigger, customer), identify the SINGLE most critical signal for the next message.

Output ONLY this JSON — no markdown, no explanation:
{"signal": "<best_signal_type>", "signal_detail": "<why this signal matters now>", "best_offer": "<offer TITLE to pair (NEVER use ID), or null>", "key_fact": "<the ONE specific number/fact to anchor the message>", "merchant_name": "<name>", "owner_name": "<owner first name>"}"""


def _run_diagnostician(context_bundle: str) -> Dict[str, Any]:
    """Call Cerebras llama3.1-8b to extract the critical signal."""
    if not CEREBRAS_API_KEY:
        logger.warning("CEREBRAS_API_KEY not set — using heuristic signal extraction")
        return {}

    try:
        resp = requests.post(CEREBRAS_URL, headers={
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": DIAG_MODEL,
            "messages": [
                {"role": "system", "content": DIAG_SYSTEM},
                {"role": "user", "content": context_bundle},
            ],
            "temperature": 0.0, "max_tokens": 400, "top_p": 1.0,
        }, timeout=15)

        if resp.status_code != 200:
            logger.error("Cerebras %d: %s", resp.status_code, resp.text[:300])
            return {}

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_json(raw) or {}
    except Exception as e:
        logger.error("Diagnostician error: %s", e)
        return {}


# ─── Step 2: Copywriter (Groq llama-3.3-70b-versatile) ───────────────────────

def _build_copy_prompt(signal: Dict, context_bundle: str, is_reply: bool = False, reply_msg: str = "", conv_history: str = "", from_role: str = "merchant") -> str:
    parts = []
    if is_reply:
        parts.append(f'The {from_role} replied: "{reply_msg}"')
        if conv_history:
            parts.append(f"Conversation so far:\n{conv_history}")
        parts.append("Craft a contextual reply. Acknowledge their message briefly, then pivot immediately to the signal.")
        parts.append("CRITICAL: You MUST use exact numbers, offer titles, or metrics from the Full Context below in your response. Generic replies score 0.")
        parts.append("If they said 'not interested' or hostile → action should be 'end'.")
        parts.append("If it's an auto-reply (canned 'thank you for contacting') → action should be 'wait' with wait_seconds.")
        parts.append("If they committed or confirmed → switch to ACTION mode with concrete next steps, not more questions.")
        parts.append("If off-topic (GST, unrelated) → politely decline and redirect to your signal.")
        if from_role == "customer":
            parts.append("IMPORTANT: You are replying to a CUSTOMER on behalf of the merchant. Use the customer's name, NOT the merchant owner's name. Confirm their request specifically.")
    else:
        parts.append("Craft a proactive outreach message.")
        parts.append("CRITICAL: You MUST include a specific metric (e.g., footfall, views, churn rate) or an exact offer title from the data. Generic messages score 0.")
        parts.append("High compulsion. Keep it under 3 sentences. ONE clear CTA.")
    
    parts.append(f"\nSignal: {json.dumps(signal)}")
    parts.append(f"\nFull context:\n{context_bundle[:6000]}")

    if is_reply:
        parts.append("""
Output ONLY this JSON:
{"action": "send|wait|end", "body": "<message>", "cta": "<open_ended|binary_yes_no|multi_choice_slot|binary_confirm_cancel|none>", "rationale": "<why>", "wait_seconds": <int or null>}

Rules:
- "send": you have a message to send
- "wait": back off (set wait_seconds)  
- "end": close conversation gracefully
- body: specific, grounded, no fabricated facts, no URLs
- If action is "end" or "wait", body can be empty or a short closing line""")
    else:
        parts.append("""
Output ONLY this JSON:
{"body": "<message text>", "cta": "<open_ended|binary_yes_no|multi_choice_slot|binary_confirm_cancel>", "send_as": "<vera|merchant_on_behalf>", "rationale": "<1-2 sentences>", "template_name": "<short_template_id>", "template_params": ["<param1>", "<param2>"]}

Rules:
- body MUST reference specific numbers/offers/facts from the context
- NEVER fabricate data not in the context
- NO URLs in body
- One clear CTA
- send_as: use "merchant_on_behalf" for customer-scoped triggers, "vera" otherwise""")

    return "\n".join(parts)


def _run_copywriter(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call Groq llama-3.3-70b-versatile to draft the message."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback message")
        return {}

    try:
        resp = requests.post(GROQ_URL, headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": COPY_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0, "max_tokens": 1024, "top_p": 1.0,
        }, timeout=20)

        if resp.status_code != 200:
            logger.error("Groq Copywriter %d: %s", resp.status_code, resp.text[:300])
            return {}

        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_json(raw) or {}
    except Exception as e:
        logger.error("Copywriter error: %s", e)
        return {}


# ─── Context Bundle Builder ──────────────────────────────────────────────────

def build_context_bundle(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict] = None) -> str:
    """Serialize the 4-context framework into a single text block for LLM input."""
    parts = []
    parts.append(f"=== CATEGORY ({category.get('slug','?')}) ===")
    voice = category.get("voice", {})
    parts.append(f"Tone: {voice.get('tone','?')}")
    taboos = voice.get("vocab_taboo", voice.get("taboos", []))
    if taboos:
        parts.append(f"Taboos: {taboos}")
    digest = category.get("digest", [])
    if digest:
        parts.append(f"Digest items: {json.dumps(digest[:3])}")
    offers_cat = category.get("offer_catalog", [])
    if offers_cat:
        parts.append(f"Category offers: {json.dumps(offers_cat[:4])}")
    peers = category.get("peer_stats", {})
    if peers:
        parts.append(f"Peer stats: {json.dumps(peers)}")
    seasonal = category.get("seasonal_beats", [])
    if seasonal:
        parts.append(f"Seasonal: {json.dumps(seasonal[:3])}")
    trends = category.get("trend_signals", [])
    if trends:
        parts.append(f"Trends: {json.dumps(trends[:3])}")

    parts.append(f"\n=== MERCHANT ({merchant.get('merchant_id','?')}) ===")
    ident = merchant.get("identity", {})
    parts.append(f"Name: {ident.get('name','?')}, Owner: {ident.get('owner_first_name','?')}")
    parts.append(f"City: {ident.get('city','?')}, Locality: {ident.get('locality','?')}")
    parts.append(f"Languages: {ident.get('languages',[])}")
    perf = merchant.get("performance", {})
    if perf:
        parts.append(f"Performance (30d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, ctr={perf.get('ctr','?')}, directions={perf.get('directions','?')}")
        delta = perf.get("delta_7d", {})
        if delta:
            parts.append(f"7d delta: {json.dumps(delta)}")
    signals = merchant.get("signals", [])
    if signals:
        parts.append(f"Signals: {signals}")
    m_offers = merchant.get("offers", [])
    if m_offers:
        active = [o for o in m_offers if o.get("status") == "active"]
        parts.append(f"Active offers: {json.dumps(active)}")
    conv_hist = merchant.get("conversation_history", [])
    if conv_hist:
        parts.append(f"Conversation history: {json.dumps(conv_hist[-3:])}")
    cust_agg = merchant.get("customer_aggregate", {})
    if cust_agg:
        parts.append(f"Customer aggregate: {json.dumps(cust_agg)}")
    reviews = merchant.get("review_themes", [])
    if reviews:
        parts.append(f"Review themes: {json.dumps(reviews)}")
    sub = merchant.get("subscription", {})
    if sub:
        parts.append(f"Subscription: {json.dumps(sub)}")

    parts.append(f"\n=== TRIGGER ({trigger.get('id','?')}) ===")
    parts.append(f"Kind: {trigger.get('kind','?')}, Urgency: {trigger.get('urgency','?')}")
    parts.append(f"Scope: {trigger.get('scope','?')}")
    trig_payload = trigger.get("payload", {})
    if trig_payload:
        parts.append(f"Payload: {json.dumps(trig_payload)}")
    parts.append(f"Suppression key: {trigger.get('suppression_key','')}")

    if customer:
        parts.append(f"\n=== CUSTOMER ({customer.get('customer_id','?')}) ===")
        c_ident = customer.get("identity", {})
        parts.append(f"Name: {c_ident.get('name','?')}, Lang: {c_ident.get('language_pref','?')}")
        rel = customer.get("relationship", {})
        if rel:
            parts.append(f"Visits: {rel.get('visits_total','?')}, Last: {rel.get('last_visit','?')}, Services: {rel.get('services_received',[])[: 5]}")
        parts.append(f"State: {customer.get('state','?')}")
        prefs = customer.get("preferences", {})
        if prefs:
            parts.append(f"Preferences: {json.dumps(prefs)}")
        consent = customer.get("consent", {})
        if consent:
            parts.append(f"Consent scope: {consent.get('scope',[])}")

    return "\n".join(parts)


# ─── Compose for /v1/tick ─────────────────────────────────────────────────────

def compose_tick_action(
    category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict] = None
) -> Dict[str, Any]:
    """Full compose pipeline for a tick action. Returns the action dict."""
    cat_slug = merchant.get("category_slug", category.get("slug", "general"))
    bundle = build_context_bundle(category, merchant, trigger, customer)

    # Step 1: Diagnostician
    signal = _run_diagnostician(bundle)
    if not signal:
        signal = _heuristic_signal(merchant, trigger, customer)

    # Step 2: Copywriter
    voice = _get_voice(cat_slug, category)
    user_prompt = _build_copy_prompt(signal, bundle, is_reply=False)
    result = _run_copywriter(voice, user_prompt)

    if not result or not result.get("body"):
        result = _fallback_tick_message(signal, merchant, trigger, customer, category)

    # Build the full action envelope
    mid = merchant.get("merchant_id", "")
    cid = customer.get("customer_id") if customer else trigger.get("customer_id")
    tid = trigger.get("id", "")
    conv_id = f"conv_{mid}_{tid}" if tid else f"conv_{mid}_{uuid.uuid4().hex[:8]}"

    return {
        "conversation_id": conv_id,
        "merchant_id": mid,
        "customer_id": cid,
        "send_as": result.get("send_as", "merchant_on_behalf" if cid else "vera"),
        "trigger_id": tid,
        "template_name": result.get("template_name", f"vera_{trigger.get('kind','generic')}_v1"),
        "template_params": result.get("template_params", []),
        "body": result.get("body", ""),
        "cta": result.get("cta", "open_ended"),
        "suppression_key": trigger.get("suppression_key", f"auto:{mid}:{tid}"),
        "rationale": result.get("rationale", "Composed from category+merchant+trigger context"),
    }


# ─── Compose for /v1/reply ───────────────────────────────────────────────────

def compose_reply(
    merchant: Dict, category: Dict, message: str,
    conversation_history: List[Dict] = None,
    trigger: Optional[Dict] = None, customer: Optional[Dict] = None,
    from_role: str = "merchant",
) -> Dict[str, Any]:
    """Full compose pipeline for a reply. Returns {action, body, cta, rationale}."""
    cat_slug = merchant.get("category_slug", category.get("slug", "general"))
    
    # Build conversation history string
    conv_str = ""
    if conversation_history:
        conv_parts = []
        for t in conversation_history[-6:]:
            conv_parts.append(f"[{t['role']}]: {t['message']}")
        conv_str = "\n".join(conv_parts)

    # Use a minimal trigger if none provided
    if not trigger:
        trigger = {"id": "reply_context", "kind": "reply", "scope": "merchant",
                   "payload": {}, "urgency": 2, "suppression_key": ""}

    bundle = build_context_bundle(category, merchant, trigger, customer)

    # Step 1: Diagnostician  
    signal = _run_diagnostician(bundle)
    if not signal:
        signal = _heuristic_signal(merchant, trigger, customer)

    # Check deterministic heuristics first to bypass LLM for strict rules
    heuristic_result = _fallback_reply(message, merchant, signal, conversation_history, customer, from_role)
    
    if heuristic_result.get("action") in ["end"] or heuristic_result.get("wait_seconds") == 14400:
        result = heuristic_result
    else:
        # Step 2: Copywriter (reply mode)
        voice = _get_voice(cat_slug, category)
        user_prompt = _build_copy_prompt(signal, bundle, is_reply=True, reply_msg=message, conv_history=conv_str, from_role=from_role)
        result = _run_copywriter(voice, user_prompt)
    
        if not result:
            result = heuristic_result

    # Ensure required fields
    result.setdefault("action", "send")
    result.setdefault("body", "")
    result.setdefault("cta", "open_ended")
    result.setdefault("rationale", "Reply composed from context")

    # Clean up wait_seconds
    if result["action"] == "wait" and not result.get("wait_seconds"):
        result["wait_seconds"] = 3600

    return result


# ─── Heuristic Fallbacks ─────────────────────────────────────────────────────

def _heuristic_signal(merchant: Dict, trigger: Dict, customer: Optional[Dict] = None) -> Dict:
    """Deterministic signal extraction when Cerebras is unavailable."""
    ident = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    
    return {
        "signal": trigger.get("kind", "general_engagement"),
        "signal_detail": json.dumps(trigger.get("payload", {}))[:200],
        "best_offer": active_offers[0].get("title") if active_offers else None,
        "key_fact": f"views={perf.get('views','?')}, calls={perf.get('calls','?')}" if perf else "",
        "merchant_name": ident.get("name", ""),
        "owner_name": ident.get("owner_first_name", ""),
    }


def _fallback_tick_message(signal: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict], category: Dict) -> Dict:
    """Grounded fallback when Groq is unavailable."""
    ident = merchant.get("identity", {})
    name = ident.get("owner_first_name", ident.get("name", ""))
    perf = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    active = [o for o in offers if o.get("status") == "active"]
    kind = trigger.get("kind", "update")
    trig_payload = trigger.get("payload", {})
    cat_slug = category.get("slug", "")

    # Build a grounded message based on trigger kind
    if customer:
        c_name = customer.get("identity", {}).get("name", "Customer")
        body = f"Hi {c_name}, {ident.get('name','')} here. "
        if kind == "recall_due":
            slots = trig_payload.get("available_slots", [])
            slot_str = " or ".join([s.get("label","") for s in slots[:2]]) if slots else "this week"
            body += f"Your {trig_payload.get('service_due','check-up')} is due. Available: {slot_str}."
            if active:
                body += f" {active[0]['title']} included."
        elif kind == "chronic_refill_due":
            mols = trig_payload.get("molecule_list", [])
            body += f"Your refill for {', '.join(mols[:3])} is coming up. "
            body += "Shall I arrange home delivery?"
        else:
            body += f"Following up on your recent visit. We have an update for you."
        return {"body": body, "cta": "binary_yes_no", "send_as": "merchant_on_behalf",
                "rationale": f"Customer-scoped {kind} trigger", "template_name": f"vera_{kind}_v1", "template_params": [c_name, name]}
    
    # Merchant-scoped
    greeting = f"Dr. {name}" if cat_slug == "dentists" else name
    body = f"{greeting}, "
    
    if kind == "research_digest":
        digest = category.get("digest", [])
        top_id = trig_payload.get("top_item_id", "")
        item = next((d for d in digest if d.get("id") == top_id), None)
        if item:
            body += f"{item.get('source','New research')} — {item['title']}. {item.get('summary','')[:100]}"
            body += f" Worth a look?"
        else:
            body += "New research relevant to your practice just landed. Want me to pull the details?"
    elif kind == "perf_dip" or kind == "seasonal_perf_dip":
        metric = trig_payload.get("metric", "views")
        delta = trig_payload.get("delta_pct", 0)
        body += f"Your {metric} dipped {abs(int(delta*100))}% this week. "
        if active:
            body += f"Your '{active[0]['title']}' offer could help recover. Want me to push it?"
        else:
            body += "Want me to draft an offer to bring traffic back?"
    elif kind == "renewal_due":
        days = trig_payload.get("days_remaining", "?")
        body += f"Your Pro subscription expires in {days} days. "
        body += f"Current performance: {perf.get('views','?')} views, {perf.get('calls','?')} calls this month. Renew to keep the momentum?"
    elif kind == "ipl_match_today":
        match = trig_payload.get("match", "tonight's match")
        body += f"{match} is tonight! Your locality gets high footfall on match nights. "
        if active:
            body += f"Want me to push your '{active[0]['title']}' as a match-night special?"
        else:
            body += "Want to run a match-night special?"
    else:
        body += f"Quick update based on your latest data — {perf.get('views','?')} views this month"
        if active:
            body += f", and your '{active[0]['title']}' is live"
        body += ". Want me to help optimize?"
    
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": f"Signal: {kind}. Grounded in merchant perf + trigger payload.",
            "template_name": f"vera_{kind}_v1", "template_params": [greeting]}


def _fallback_reply(message: str, merchant: Dict, signal: Dict, conversation_history: List[Dict] = None, customer: Optional[Dict] = None, from_role: str = "merchant") -> Dict:
    """Deterministic fallback for reply composition. Role-aware."""
    msg_lower = message.lower().strip()
    
    # Auto-reply detection
    auto_phrases = ["thank you for contacting", "our team will respond", "we will get back", "auto-reply", "out of office", "away message"]
    if any(p in msg_lower for p in auto_phrases):
        auto_count = 1
        if conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "merchant" and any(p in turn.get("message", "").lower() for p in auto_phrases):
                    auto_count += 1
                elif turn.get("role") == "merchant":
                    break
        ident = merchant.get("identity", {})
        m_name = ident.get("name", "your business")
        o_name = ident.get("owner_first_name", "there")
        
        if auto_count >= 4:
            body = f"Since we haven't been able to connect, {o_name}, I'll close this thread for {m_name}. Reach out when you're ready!"
            return {"action": "end", "body": body, "rationale": "Auto-reply limit reached."}
        
        body = f"Understood, {o_name}. Since {m_name} is currently unavailable, we will pause our performance outreach for now."
        return {"action": "wait", "wait_seconds": 14400, "body": body, "cta": "none", "rationale": f"Detected auto-reply (count: {auto_count})."}
    
    # Hostile detection
    hostile_phrases = ["stop messaging", "not interested", "useless", "spam", "stop sending",
                       "don't message", "unsubscribe", "leave me alone"]
    if any(p in msg_lower for p in hostile_phrases):
        ident = merchant.get("identity", {})
        m_name = ident.get("name", "your business")
        o_name = ident.get("owner_first_name", "there")
        
        body = f"Noted, {o_name}. We have updated the preferences for {m_name} and will halt all performance-driven messaging immediately."
        return {"action": "end", "body": body, "rationale": "User explicitly opted out. Closing conversation gracefully."}
    
    # Commitment / booking detection → branch by who is speaking
    commit_phrases = ["ok let's do it", "let's do it", "yes do it", "go ahead", "proceed",
                      "sounds good let's", "ok go ahead", "yes please", "confirm", "let's go",
                      "whats next", "what's next", "book me", "yes please book", "sign me up"]
    if any(p in msg_lower for p in commit_phrases):
        if from_role == "customer" or customer:
            # CUSTOMER is confirming — use customer name, confirm their slot/action
            c_name = "there"
            if customer:
                c_name = customer.get("identity", {}).get("name", "there")
            
            ident = merchant.get("identity", {})
            m_name = ident.get("name", "our clinic")
            
            body = f"Perfect, {c_name}! Your slot at {m_name} is confirmed. "
            best_off = signal.get("best_offer", "")
            if best_off:
                body += f"Don't forget you can use the '{best_off}' offer during your visit! "
            body += "We look forward to serving you."
            
            return {"action": "send",
                    "body": body,
                    "cta": "none",
                    "rationale": "Customer committed — confirming appointment/action."}
        else:
            # MERCHANT is committing — switch to action mode
            ident = merchant.get("identity", {})
            name = ident.get("owner_first_name", ident.get("name", ""))
            
            body = f"Great, {name}! Setting this up now. "
            best_off = signal.get("best_offer", "")
            if best_off:
                body += f"We'll highlight your '{best_off}' offer to drive more engagement. "
            body += "I'll have the draft ready in a moment. You'll be able to review before anything goes live."
            
            return {"action": "send",
                    "body": body,
                    "cta": "binary_confirm_cancel",
                    "rationale": "Merchant committed — switching from qualifying to action mode."}
    
    # Off-topic detection
    offtopic = ["gst", "tax", "invoice", "salary", "loan", "insurance"]
    if any(w in msg_lower for w in offtopic):
        return {"action": "send",
                "body": "That's outside what I can help with directly — best to check with your CA on that. Coming back to your business growth — want me to continue with what we were working on?",
                "cta": "open_ended",
                "rationale": "Off-topic ask declined politely; redirecting to growth signal."}
    
    # Default engaged reply — grounded in signal data, not generic
    ident = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    active = [o for o in offers if o.get("status") == "active"]
    key_fact = signal.get("key_fact", "")
    best_offer = signal.get("best_offer", "")
    
    if from_role == "customer" and customer:
        c_name = customer.get("identity", {}).get("name", "there")
        body = f"Thanks for reaching out, {c_name}! "
        if best_offer:
            body += f"We have '{best_offer}' available for you. "
        body += "Would you like to book a slot?"
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": f"Customer replied — offering specific action grounded in {signal.get('signal','context')}."}
    else:
        name = ident.get("owner_first_name", ident.get("name", ""))
        body = f"Noted, {name}. "
        if key_fact:
            body += f"Looking at your numbers ({key_fact}), "
        if best_offer:
            body += f"I can pair this with your '{best_offer}' offer. "
        body += "Want me to put together a specific plan?"
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": f"Acknowledged merchant reply; advancing with grounded data from {signal.get('signal','context')}."}


def _parse_json(text: str) -> Optional[Dict]:
    """Robustly parse JSON from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except json.JSONDecodeError:
            pass
    return None
