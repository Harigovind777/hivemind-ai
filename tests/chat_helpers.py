"""A scripted team for chat tests: drafts the toy spec, reads follow-ups, answers questions."""

from hivemind.agents import Team
from hivemind.coordinator import Coordinator
from hivemind.llm import Ledger, ScriptedLLM

from test_coordinator import TASK, respond

SPEC = TASK.prompt  # def add_one(x: int) -> int, with one >>> example
CHANGED_SPEC = SPEC.replace("Return x plus one.", "Return x plus one, or 0 for None.")


def chat_respond(system: str, prompt: str) -> str:
    if system.startswith("You are the Research") and "Do not write the function body" in prompt:
        spec = CHANGED_SPEC if "The user asked for this change" in prompt else SPEC
        return f"```python\n{spec}```"
    if system.startswith("You are the Coordinator") and "Reply with one word" in prompt:
        message = prompt.split("The user's new message:")[1]
        return "CHANGE" if "please also" in message else "QUESTION"
    if system.startswith("You are the Coordinator") and "The user's question" in prompt:
        return "Candidate 1 matched every verified case."
    return respond(system, prompt)


def chat_coordinator(respond_fn=chat_respond):
    llm = ScriptedLLM(respond_fn)
    return Coordinator(Team(llm, llm, Ledger()), n_candidates=3)
