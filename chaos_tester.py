#!/usr/bin/env python3
"""
Vera Message Engine - Chaos & Edge Case Tester
Injects adversarial payloads (Customer Replies, Hostility, Off-Topic) 
and scores them using the official magicpin LLMScorer.
"""

import os
import json
import time
from datetime import datetime, timezone
from urllib import request as urlrequest

# Import the official magicpin scoring engine to guarantee unbiased results
from judge_simulator import create_provider, LLMScorer, DatasetLoader, DATASET_DIR
from judge_simulator import print_header, print_section, print_info, print_success, print_fail, Colors, print_score_bar, print_reason

# Point this to your local server to test the deployed code before pushing
BOT_URL = "http://localhost:8000"

# Adversarial Edge Cases
CHAOS_SCENARIOS = [
    {
        "name": "Customer Typoglycemia (Misspelled booking)",
        "payload": {
            "conversation_id": "conv_chaos_1",
            "merchant_id": "m_001_drmeera",
            "customer_id": "c_001_test",
            "from_role": "customer",
            "message": "yesss plz b00k me for tmrw 5pm!!",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "turn_number": 2
        }
    },
    {
        "name": "Merchant Off-Topic (Tax question)",
        "payload": {
            "conversation_id": "conv_chaos_2",
            "merchant_id": "m_001_drmeera",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Hey Vera, what is the new GST rate for dental implants?",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "turn_number": 2
        }
    },
    {
        "name": "Customer Aggressive / Hostile",
        "payload": {
            "conversation_id": "conv_chaos_3",
            "merchant_id": "m_001_drmeera",
            "customer_id": "c_001_test",
            "from_role": "customer",
            "message": "STOP SPAMMING MY PHONE OR I WILL SUE",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "turn_number": 2
        }
    },
    {
        "name": "Merchant Conflicting Command",
        "payload": {
            "conversation_id": "conv_chaos_4",
            "merchant_id": "m_001_drmeera",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Actually don't run the discount, make it free, no wait cancel the whole thing.",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "turn_number": 2
        }
    }
]

def send_custom_reply(payload: dict) -> dict:
    """Sends a raw HTTP request to bypass the limited judge_simulator BotClient."""
    url = f"{BOT_URL}/v1/reply"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        resp = urlrequest.urlopen(req, timeout=15)
        return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print_fail(f"Request failed: {e}")
        return {}

def run_chaos():
    print_header("VERA MESSAGE ENGINE — CHAOS TESTER")
    
    # 1. Load the official scorer and dataset
    llm = create_provider()
    dataset = DatasetLoader(DATASET_DIR)
    dataset.load()
    scorer = LLMScorer(llm, dataset)
    print_info(f"Loaded Official Scorer via {llm.name()}")

    # 2. Extract base context for scoring (using Dr. Meera as the anchor)
    merchant = dataset.merchants.get("m_001_drmeera", {})
    category = dataset.categories.get(merchant.get("category_slug", "dentists"), {})
    customer = dataset.customers.get("c_001_test", {"identity": {"name": "Test Customer"}})
    trigger = {"kind": "chaos_test", "payload": {}, "urgency": 5}

    print_section("EXECUTING CHAOS PAYLOADS")

    total_score = 0
    max_possible = len(CHAOS_SCENARIOS) * 50

    for idx, scenario in enumerate(CHAOS_SCENARIOS):
        print(f"\n{Colors.CYAN}{Colors.BOLD}Test {idx+1}: {scenario['name']}{Colors.RESET}")
        print(f"{Colors.DIM}Sender: {scenario['payload']['from_role'].upper()} | Message: \"{scenario['payload']['message']}\"{Colors.RESET}")
        
        # Hit the bot
        start = time.time()
        bot_response = send_custom_reply(scenario["payload"])
        latency = (time.time() - start) * 1000

        if not bot_response:
            continue

        print_info(f"Bot Action: {bot_response.get('action')} ({latency:.0f}ms)")
        print(f"{Colors.MAGENTA}Bot Body: \"{bot_response.get('body', '')}\"{Colors.RESET}")

        # Score the response using official magicpin logic
        score = scorer.score(bot_response, category, merchant, trigger, customer if scenario["payload"]["from_role"] == "customer" else None)
        
        print_score_bar("Specificity", score.specificity)
        print_score_bar("Category Fit", score.category_fit)
        print_score_bar("Merchant Fit", score.merchant_fit)
        print_score_bar("Decision Quality", score.decision_quality)
        print_score_bar("Engagement", score.engagement_compulsion)
        print(f"  {Colors.BOLD}Scenario Total: {score.total}/50{Colors.RESET}")
        
        total_score += score.total

    print_section("CHAOS TEST SUMMARY")
    pct = (total_score / max_possible) * 100
    print(f"{Colors.BOLD}FINAL CHAOS SCORE: {total_score}/{max_possible} ({pct:.0f}%){Colors.RESET}")
    if pct >= 80:
        print_success("Bot is highly resilient to adversarial edge cases.")
    else:
        print_fail("Bot requires further heuristic grounding for edge cases.")

if __name__ == "__main__":
    run_chaos()