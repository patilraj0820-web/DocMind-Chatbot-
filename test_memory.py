"""
test_memory.py
--------------
Run this script (with the server already running) to verify that
ConversationBufferWindowMemory is working correctly.

Usage:
    python3 test_memory.py                         # uses default document
    python3 test_memory.py --upload myfile.pdf     # upload a file first
"""

import sys
import json
import argparse
import requests

BASE_URL = "http://localhost:8000"
SESSION_ID = "memory-test-session-001"


def print_separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def upload_file(filepath: str) -> None:
    print_separator(f"Uploading: {filepath}")
    with open(filepath, "rb") as f:
        resp = requests.post(f"{BASE_URL}/upload", files={"file": f}, timeout=60)
    resp.raise_for_status()
    print(f"✅ Upload response: {json.dumps(resp.json(), indent=2)}")


def ask(question: str, turn: int) -> str:
    print_separator(f"Turn {turn} — Question")
    print(f"  ❓ {question}")

    payload = {
        "question": question,
        "session_id": SESSION_ID,
        "model": "qwen2.5:3b",
    }
    resp = requests.post(f"{BASE_URL}/rag/ask", json=payload, timeout=120)
    resp.raise_for_status()
    answer = resp.json().get("answer", "")
    print(f"\n  🤖 Answer:\n{answer}")
    return answer


def check_memory_directly() -> None:
    """
    Directly instantiate the memory module to inspect the buffer
    without going through the HTTP server.
    """
    print_separator("Direct memory buffer inspection")
    try:
        # Add project root to path so we can import backend
        sys.path.insert(0, ".")
        from backend.memory import get_history_text
        history = get_history_text(SESSION_ID)
        if history.strip():
            print("✅ Memory buffer contains:\n")
            print(history)
        else:
            print("⚠️  Memory buffer is EMPTY — memory may not be saving turns.")
    except Exception as e:
        print(f"⚠️  Could not inspect memory directly: {e}")
        print("    (This is OK if testing via HTTP only.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", help="Path to a document to upload before testing")
    args = parser.parse_args()

    print("\n🧪 RAG Chatbot — Conversation Memory Test")
    print(f"   Base URL  : {BASE_URL}")
    print(f"   Session ID: {SESSION_ID}")

    # ── Optional upload ───────────────────────────────────────────
    if args.upload:
        upload_file(args.upload)

    # ── Turn 1: General question ──────────────────────────────────
    answer1 = ask("What is this document about?", turn=1)

    # ── Turn 2: Follow-up that only makes sense with memory ───────
    answer2 = ask(
        "Based on what you just told me, what are the key points I should remember?",
        turn=2,
    )

    # ── Turn 3: Explicit memory reference ────────────────────────
    answer3 = ask(
        "What was my very first question in this conversation?",
        turn=3,
    )

    # ── Inspect the buffer directly ───────────────────────────────
    check_memory_directly()

    # ── Summary ──────────────────────────────────────────────────
    print_separator("Test Summary")
    memory_hint_words = ["you asked", "first question", "earlier", "mentioned", "previously"]
    memory_found = any(w.lower() in answer3.lower() for w in memory_hint_words)

    if memory_found:
        print("✅ PASS — Turn 3 answer references prior conversation context.")
    else:
        print("⚠️  Turn 3 answer did NOT explicitly reference prior context.")
        print("   This could still be OK depending on the document content.")
        print("   Check the memory buffer above for saved turns.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
