#!/usr/bin/env python3
import os
# Force the judge to use the thinking model for exhaustive testing
os.environ["COPY_MODEL"] = "qwen/qwen3-32b"

import time, json
from datetime import datetime, timezone
from urllib import request as urlrequest
from judge_simulator import create_provider, LLMScorer, DatasetLoader, DATASET_DIR
from judge_simulator import print_header, print_section, print_info, print_score_bar, Colors, print_fail

BOT_URL = "http://localhost:8000"

INTENT_MATRIX = [
    {"name": "Customer Booking", "role": "customer", "msg": "Yes please book me for Wed 5 Nov, 6pm."},
    {"name": "Merchant Commitment", "role": "merchant", "msg": "Sounds good let's proceed."},
    {"name": "Merchant Objection", "role": "merchant", "msg": "I don't want to run a discount, it hurts my brand."},
    {"name": "Hostile/Stop", "role": "merchant", "msg": "stop sending me these messages immediately"},
    {"name": "Auto-Reply Simulator", "role": "merchant", "msg": "Thank you for contacting us. We will get back to you."}
]

def send_request(endpoint: str, payload: dict) -> dict:
    url = f"{BOT_URL}{endpoint}"
    req = urlrequest.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
    try:
        return json.loads(urlrequest.urlopen(req, timeout=60).read().decode("utf-8"))
    except Exception as e:
        print_fail(f"Request failed to {endpoint}: {e}")
        return {}

def run_exhaustive():
    print_header("VERA MESSAGE ENGINE — EXHAUSTIVE MATRIX TESTER")
    
    llm = create_provider()
    dataset = DatasetLoader(DATASET_DIR)
    dataset.load()
    scorer = LLMScorer(llm, dataset)
    
    test_merchants = {}
    for mid, m_data in dataset.merchants.items():
        cat = m_data.get("category_slug")
        if cat and cat not in test_merchants:
            test_merchants[cat] = mid
        if len(test_merchants) == 5: break

    total_score, max_possible = 0, 0
    dimension_totals = {"Specificity": 0, "Category Fit": 0, "Merchant Fit": 0, "Decision Quality": 0, "Engagement": 0}

    print_section("EXECUTING EXHAUSTIVE MATRIX")
    
    # Open log file
    log_file_path = "exhaustive_test_logs.log"
    with open(log_file_path, "w", encoding="utf-8") as logf:
        logf.write("=== EXHAUSTIVE TEST LOGS ===\n\n")

    for cat_slug, mid in test_merchants.items():
        merchant = dataset.merchants[mid]
        category = dataset.categories.get(cat_slug, {})
        customer = list(dataset.customers.values())[0]
        
        # FIX 1: Seed the bot's database BEFORE testing replies (Use timestamp version to prevent 409 Conflict)
        v = int(time.time())
        send_request("/v1/context", {"scope": "category", "context_id": cat_slug, "version": v, "payload": category})
        send_request("/v1/context", {"scope": "merchant", "context_id": mid, "version": v, "payload": merchant})
        send_request("/v1/context", {"scope": "customer", "context_id": customer["customer_id"], "version": v, "payload": customer})

        for intent in INTENT_MATRIX:
            print(f"\n{Colors.CYAN}{Colors.BOLD}[{cat_slug.upper()}] - {intent['name']}{Colors.RESET}")
            
            payload = {
                "conversation_id": f"conv_{mid}_{intent['name'][:5]}",
                "merchant_id": mid,
                "customer_id": customer["customer_id"] if intent["role"] == "customer" else None,
                "from_role": intent["role"],
                "message": intent["msg"],
                "received_at": datetime.now(timezone.utc).isoformat(),
                "turn_number": 2
            }
            
            bot_response = send_request("/v1/reply", payload)
            if not bot_response: continue

            print(f"{Colors.MAGENTA}Bot Action: {bot_response.get('action')} | Body: {bot_response.get('body', '')[:60]}...{Colors.RESET}")
            
            score = scorer.score(bot_response, category, merchant, {"kind": "test", "payload": {}}, customer if intent["role"] == "customer" else None)
            
            # Log to file
            with open(log_file_path, "a", encoding="utf-8") as logf:
                logf.write(f"[{cat_slug.upper()}] - {intent['name']}\n")
                logf.write(f"User Message ({intent['role']}): \"{intent['msg']}\"\n")
                logf.write(f"Bot Action: {bot_response.get('action')}\n")
                logf.write(f"Bot Body: {bot_response.get('body', '')}\n")
                logf.write(f"Scores:\n")
                logf.write(f"  Specificity: {score.specificity}/10 - {score.specificity_reason}\n")
                logf.write(f"  Category Fit: {score.category_fit}/10 - {score.category_fit_reason}\n")
                logf.write(f"  Merchant Fit: {score.merchant_fit}/10 - {score.merchant_fit_reason}\n")
                logf.write(f"  Decision Quality: {score.decision_quality}/10 - {score.decision_quality_reason}\n")
                logf.write(f"  Engagement: {score.engagement_compulsion}/10 - {score.engagement_reason}\n")
                logf.write(f"  Total: {score.total}/50\n")
                logf.write("-" * 50 + "\n\n")

            dimension_totals["Specificity"] += score.specificity
            dimension_totals["Category Fit"] += score.category_fit
            dimension_totals["Merchant Fit"] += score.merchant_fit
            dimension_totals["Decision Quality"] += score.decision_quality
            dimension_totals["Engagement"] += score.engagement_compulsion
            
            total_score += score.total
            max_possible += 50

            # FIX 2: Protect the Groq API limits (6K TPM for Qwen, 12K TPM for Llama 70B)
            # Sleep for 16 seconds to ensure we do not exceed the Tokens Per Minute limit
            time.sleep(16)

    print_section("FINAL SCORECARD PREDICTION")
    runs = len(test_merchants) * len(INTENT_MATRIX)
    print(f"Specificity:           {dimension_totals['Specificity'] / runs:.1f} / 10")
    print(f"Category Fit:          {dimension_totals['Category Fit'] / runs:.1f} / 10")
    print(f"Merchant Fit:          {dimension_totals['Merchant Fit'] / runs:.1f} / 10")
    print(f"Decision Quality:      {dimension_totals['Decision Quality'] / runs:.1f} / 10")
    print(f"Engagement Compulsion: {dimension_totals['Engagement'] / runs:.1f} / 10")
    
    pct = (total_score / max_possible) * 100
    print(f"\n{Colors.BOLD}PREDICTED TOTAL SCORE: {total_score}/{max_possible} ({pct:.0f}%){Colors.RESET}")

if __name__ == "__main__":
    run_exhaustive()