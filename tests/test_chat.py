"""Chat calls: spec drafting and checking, reading follow-ups, answering questions."""

import ast

import pytest

from hivemind.agents import Team, Transcript
from hivemind.chat import (ModelCallFailed, SpecError, answer_question, classify, custom_task,
                           display_source, draft_spec, round_context, spec_text, validate_spec)
from hivemind.llm import Ledger, ScriptedLLM

SPEC = ('from typing import List\n\n\ndef second_largest(nums: List[int]) -> int:\n'
        '    """Return the second largest distinct value in nums.\n'
        '    >>> second_largest([3, 1, 4, 1, 5])\n    4\n'
        '    >>> second_largest([2, 2, 1])\n    1\n    """\n')

ANSWER = {"code": "def second_largest(nums):\n    return sorted(set(nums))[-2]\n",
          "source": "candidate 1", "repaired": False, "candidates": ["a", "b", "c"],
          "distinct_behaviours": 2, "verified_passed": 3, "verified_total": 4, "spec_cases": 2,
          "ruled_cases": 1, "undisputed_cases": 1,
          "failures": [{"call": "second_largest([1])", "expected": "None", "got": "<IndexError>"}],
          "rulings": [{"call": "second_largest([5, 5])", "options": ["5", "<IndexError>"],
                       "choice": "<IndexError>", "reference_agrees": None}]}


def team(respond):
    llm = ScriptedLLM(respond)
    return Team(llm, llm, Ledger())


def drafting(replies):
    """A model that gives `replies` in order (repeating the last) and records each prompt."""
    prompts = []

    def respond(system, prompt):
        prompts.append(prompt)
        return replies[min(len(prompts), len(replies)) - 1]
    return respond, prompts


def test_a_valid_spec_keeps_imports_drops_the_stub_and_counts_examples():
    spec = validate_spec(SPEC + "    pass\n")
    assert spec.text == SPEC and spec.entry_point == "second_largest" and spec.examples == 2
    assert validate_spec("def f(x):\n\t'''Tabs are fine.'''\n    ...\n").entry_point == "f"


@pytest.mark.parametrize("text, line, words", [
    ("def f(x)\n    '''d'''\n", 1, "expected ':'"),
    ("def f(x):\n    return x\n", 1, "add a docstring"),
    ("def f(x):\n    '''d'''\n    return x\n", 3, "remove the function body"),
    ("x = 1\ndef f():\n    '''d'''\n", 1, "only imports and one function"),
    ("def f():\n    '''a'''\ndef g():\n    '''b'''\n", 3, "only one function"),
    ("def f():\n  '''two spaces'''\n", 2, "indent the docstring by 4 spaces"),
    ("def f():\n    '''d'''\nimport os\n", 3, "put imports before the function"),
    ("@cache\ndef f():\n    '''d'''\n", 1, "remove the decorator"),
    ("print('hi')", 1, "only imports and one function"),
    ("", None, "needs one function"),
])
def test_invalid_specs_say_what_to_fix_and_where(text, line, words):
    with pytest.raises(SpecError) as err:
        validate_spec(text)
    assert err.value.line == line and words in str(err.value)


def test_spec_text_takes_the_python_block_or_the_whole_reply():
    assert spec_text("Sure:\n```python\ndef f():\n    '''d'''\n```\nDone.") == "def f():\n    '''d'''"
    assert spec_text("def f():\n    '''d'''\n") == "def f():\n    '''d'''"


def test_a_draft_is_validated_and_cleaned():
    respond, prompts = drafting([f"Here it is:\n```python\n{SPEC}    pass\n```"])
    tr = Transcript("chat/test")
    draft = draft_spec(team(respond), "second largest number in a list", tr)
    assert draft.error is None and draft.text == SPEC and draft.spec.entry_point == "second_largest"
    assert "second largest number in a list" in prompts[0]
    assert [t["phase"] for t in tr.turns] == ["draft"]


def test_an_invalid_draft_gets_one_retry_with_the_error():
    respond, prompts = drafting(["```python\ndef f(x):\n    return x\n```", f"```python\n{SPEC}```"])
    draft = draft_spec(team(respond), "anything", Transcript("chat/test"))
    assert draft.spec is not None and len(prompts) == 2 and "add a docstring" in prompts[1]


def test_a_draft_still_invalid_after_the_retry_is_returned_for_editing():
    respond, prompts = drafting(["I cannot write code."])
    draft = draft_spec(team(respond), "anything", Transcript("chat/test"))
    assert draft.spec is None and draft.text == "I cannot write code." and "Line 1" in draft.error
    assert len(prompts) == 2


def test_a_change_is_drafted_from_the_current_spec():
    respond, prompts = drafting([f"```python\n{SPEC}```"])
    draft_spec(team(respond), "return None for an empty list", Transcript("chat/test"),
               current_spec=SPEC)
    assert SPEC.rstrip() in prompts[0] and "return None for an empty list" in prompts[0]
    assert "Keep everything the user did not ask to change" in prompts[0]


@pytest.mark.parametrize("reply, reading", [
    ("QUESTION", "question"), ("CHANGE", "change"), ("QUESTION or CHANGE: CHANGE", "change"),
    ("I think the user wants a change.", "question"),  # unclear: no capitalised verdict
    ("", "question"),
])
def test_classify_reads_the_last_verdict_and_defaults_to_question(reply, reading):
    tr = Transcript("chat/test")
    assert classify(team(lambda s, p: reply), "why?", "context", tr) == reading
    assert tr.turns[0]["phase"] == "classify" and tr.turns[0]["role"] == "coordinator"


def test_round_context_summarises_the_evidence():
    text = round_context(SPEC, ANSWER)
    # ChatAnswer counts candidates from 0; people (and the round's bubbles) count from 1.
    assert SPEC.rstrip() in text and "from candidate 2" in text
    assert "passes 3 of 4 verified cases (2 from the specification's examples, 1 ruled on" in text
    assert "second_largest([1]): expected None, got <IndexError>" in text
    assert "second_largest([5, 5]): options 5, <IndexError>; chose <IndexError>" in text


def test_a_question_is_answered_from_the_context_and_earlier_questions():
    seen = []
    tr = Transcript("chat/test")
    reply = answer_question(team(lambda s, p: seen.append(p) or "Because the spec says so."),
                            "why IndexError?", "CONTEXT", [("first q", "first a")], tr)
    assert reply == "Because the spec says so."
    assert "CONTEXT" in seen[0] and "Q: first q\nA: first a" in seen[0] and "why IndexError?" in seen[0]
    assert tr.turns[0]["phase"] == "question"


def test_custom_task_has_no_hidden_tests():
    task = custom_task(validate_spec(SPEC), "chat/abc/1")
    assert (task.task_id, task.entry_point, task.hidden_test, task.canonical_solution) == (
        "chat/abc/1", "second_largest", "", "")
    assert task.spec == SPEC.strip()


def failing_team():
    def respond(system, prompt):
        raise TimeoutError("timed out")
    return team(respond)


def test_a_failed_model_call_stops_the_chat_call_instead_of_looking_like_a_reply():
    tr = Transcript("chat/test")
    with pytest.raises(ModelCallFailed, match="Research agent's model call failed: timed out"):
        draft_spec(failing_team(), "anything", tr)
    assert len(tr.turns) == 1  # no retry of a call that failed
    with pytest.raises(ModelCallFailed, match="Coordinator's model call failed"):
        classify(failing_team(), "why?", "context", Transcript("chat/test"))
    with pytest.raises(ModelCallFailed):
        answer_question(failing_team(), "why?", "context", [], Transcript("chat/test"))


@pytest.mark.parametrize("odd", ["\x0c", "\x85", " ", "\x1c", "é — 😀"])
def test_a_spec_with_unusual_characters_keeps_its_closing_quotes(odd):
    text = f'def f(x):\n    """Double x. {odd} ok.\n    >>> f(2)\n    4\n    """\n'
    spec = validate_spec(text)
    assert spec.text == text and spec.examples == 1
    ast.parse(spec.text + "    pass\n")  # what the pipeline builds must still parse


def test_windows_line_endings_are_accepted():
    assert validate_spec(SPEC.replace("\n", "\r\n")).text == SPEC


def test_a_lone_surrogate_is_a_spec_error_not_a_crash():
    with pytest.raises(SpecError, match="not valid text"):
        validate_spec('def f(x):\n    """Bad \ud800 text."""\n')


def test_display_source_counts_candidates_from_one():
    assert display_source("candidate 0") == "candidate 1"
    assert display_source("repair of candidate 2") == "repair of candidate 3"
