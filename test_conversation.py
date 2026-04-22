"""
Text-based conversation tester.
Simulates a caller typing — tests GPT, function calling, and Google Calendar
without any phone number, Vobiz, or hosting.

Usage:
    python test_conversation.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.conversation import ConversationManager


async def main():
    print("=" * 55)
    print("  AI Voice Agent — Text Test (no phone needed)")
    print("  Type your message as if you are the caller.")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    agent = ConversationManager()

    # Show greeting
    greeting = agent.get_greeting()
    print(f"\n[Agent]: {greeting}\n")

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Ending test.")
            break

        print("[Agent]: ...", end="\r")
        response = await agent.process_turn(user_input)
        print(f"[Agent]: {response}\n")

        if agent.state.value == "done":
            break


if __name__ == "__main__":
    asyncio.run(main())
