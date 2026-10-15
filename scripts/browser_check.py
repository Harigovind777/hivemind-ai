"""End-to-end browser check: does a real solve render, and is every label honest?

Drives the served page with a scripted team so the whole path is exercised, including the
SSE stream and the results rendering.
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request

PROMPT = '''def second_smallest(numbers):
    """Return the second smallest distinct value.

    >>> second_smallest([4, 1, 3, 3, 2])
    2
    >>> second_smallest([9, 9, 8, 5])
    8
    """
'''

SERVER = r'''
import re, sys
sys.path.insert(0, ".")
import hivemind.webapp as web
from hivemind.llm import ScriptedLLM

BUG = "def second_smallest(numbers):\n    return sorted(set(numbers))[-2]\n"
GOOD = "def second_smallest(numbers):\n    return sorted(set(numbers))[1]\n"

def respond(system, prompt):
    if system.startswith("You are the Research"):
        return "Contract: list -> int over distinct values."
    if system.startswith("You are the Testing"):
        return ("```python\nassert second_smallest([3, 1, 2]) == 2\n"
                "assert second_smallest([9, 9, 8, 5]) == 8\n"
                "assert second_smallest([7, 2, 4, 1]) == 99\n```")
    if system.startswith("You are the Coordinator"):
        if "Disputed cases" in prompt:
            out = []
            for n, call, opts in re.findall(r"CASE (\d+): (.+)\n((?:  [A-Z]\) .+\n?)+)", prompt):
                truth = repr(eval(call, {"second_smallest": lambda n: sorted(set(n))[1]}))
                letter = next(m[0] for m in re.findall(r"([A-Z])\) (.+)", opts) if m[1] == truth)
                out.append(f"CASE {n}: {letter}")
            return "\n".join(out)
        return "Solution 1 matches the examples.\nWINNER: 1"
    if system.startswith("You are the Coding"):
        if "fails cases whose correct results" in prompt:
            return f"```python\n{GOOD}```"
        if "Your team is implementing" in prompt:
            own = prompt.split("Your current solution")[1].split("```python\n")[1].split("```")[0]
            return f"Keeping mine.\n```python\n{own}```"
        if "docstring examples by hand" in prompt or "edge cases first" in prompt:
            return f"```python\n{GOOD}```"
        return f"```python\n{BUG}```"
    if system.startswith("You are the Security"):
        return "NO ISSUES"
    if system.startswith("You are the Documentation"):
        return "Sorts the distinct values and returns the second. O(n log n) time, O(n) space."
    return "No concerns."

web.make_llm = lambda *a, **k: ScriptedLLM(respond)
app = web.create_app(default_config={
    "agents": {"provider": "scripted", "model": "s"},
    "coordinator": {"provider": "scripted", "model": "s"},
    "candidates": 3, "repair": True,
})
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8022, log_level="error")
'''


def wait_for(url: str, seconds: float = 25.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    shim = "/private/tmp/var/folders/0w/c19gq9rj2s1dfcq7q5r9knt00000gn/T/opencode/shim"
    server = subprocess.Popen([sys.executable, "-c", SERVER],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    failures = []
    try:
        if not wait_for("http://127.0.0.1:8022/"):
            print("server never came up")
            print(server.stdout.read().decode()[-2000:])
            return 1

        from playwright.async_api import async_playwright
        shots = "/private/tmp/var/folders/0w/c19gq9rj2s1dfcq7q5r9knt00000gn/T/opencode"

        async def drive():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": 1280, "height": 1200})
                console = []
                page.on("console", lambda m: console.append(f"{m.type}: {m.text}")
                        if m.type == "error" else None)
                page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

                await page.goto("http://127.0.0.1:8022/", wait_until="networkidle")
                await page.fill("#prompt", PROMPT)
                await page.click("#go")

                # The transcript streams turn by turn.
                await page.wait_for_selector(".turn:not(.thinking)", timeout=60_000)
                await page.wait_for_selector("#out .card", timeout=60_000)
                await page.wait_for_timeout(400)

                # Read everything while the page is still open; the browser is closed at
                # the end of the block, so any query after that would raise.
                turn_count = len(await page.query_selector_all(".turn"))
                cards = await page.query_selector_all("#out .card")
                card_count = len(cards)
                text = await page.inner_text("#out")
                transcript = await page.inner_text("#live")
                cfg = await page.evaluate("() => fetch('/api/config').then(r => r.json())")
                note = await page.inner_text("#cfg-note")
                body_text = await page.inner_text("body")

                await page.screenshot(path=f"{shots}/result.png", full_page=True)

                # Live transcript really streamed the agents' turns.
                if turn_count < 8:
                    failures.append(f"only {turn_count} turns streamed")
                if "…)" not in transcript and "working" in transcript.lower():
                    failures.append("transcript left a turn stuck on 'working…'")
                for role in ("RESEARCH", "CODING", "TESTING", "COORDINATOR"):
                    if role not in transcript.upper():
                        failures.append(f"transcript missing {role} turn")

                # Seven strategy cards.
                if card_count != 7:
                    failures.append(f"expected 7 strategy cards, got {card_count}")
                for name in ("solo", "vote", "debate", "tests_only",
                             "arbitration_select", "arbitration"):
                    if name not in text:
                        failures.append(f"missing strategy {name}")

                # The evidence labelling must be honest and visible.
                if "not benchmark scores" not in body_text:
                    failures.append("the not-benchmark-scores notice is missing")
                if "verified" not in text.lower():
                    failures.append("no 'verified' label on results")

                # Every strategy card must use the evidence wording. A bare "passed" here
                # would be an accuracy claim a prompt with no hidden tests cannot support.
                pills = await page.eval_on_selector_all(
                    "#out .card .pill", "els => els.map(e => e.textContent.trim())")
                if not pills:
                    failures.append("no strategy pills rendered")
                elif any(p in ("passed", "failed") for p in pills):
                    failures.append(f"strategy pills claim pass/fail: {pills}")
                elif not all("verified" in p or "no verified" in p for p in pills):
                    failures.append(f"unexpected pill wording: {pills}")

                # The Coordinator's ruling is shown with its evidence.
                if "second_smallest(" not in text:
                    failures.append("no case-level ruling table rendered")
                if "WINNER" in text:
                    failures.append("raw judge text leaked into the UI")

                # Stat labels are CSS-uppercased, so match them case-insensitively.
                for label in ("distinct behaviours", "disputed cases", "model calls",
                              "cost", "entry point", "verified cases"):
                    if label not in text.lower():
                        failures.append(f"missing stat {label!r}")

                # Config note is not "undefined".
                if "undefined" in note:
                    failures.append(f"config note shows undefined: {note!r}")
                if not cfg.get("configured"):
                    failures.append("api/config did not report configured=true")

                # Dark mode renders too.
                await page.emulate_media(color_scheme="dark")
                await page.wait_for_timeout(200)
                await page.screenshot(path=f"{shots}/result-dark.png", full_page=True)

                if console:
                    failures.append(f"console errors: {console}")
                await browser.close()

                print("turns streamed :", turn_count)
                print("strategy cards :", card_count)
                print("console errors :", console or "none")
                print("config note    :", note)

        asyncio.run(drive())
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall browser checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
