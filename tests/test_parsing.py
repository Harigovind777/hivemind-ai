from hivemind.domains.code import (doc_example_cases, extract_code, grade, load_humaneval,
                                   parse_assert_cases)
from hivemind.security import static_scan


def test_extract_code_prefers_block_defining_entry_point():
    text = ("Here is a helper:\n```python\nprint('hi')\n```\nFinal:\n"
            "```python\ndef solve(x):\n    return x\n```\n")
    assert extract_code(text, "solve") == "def solve(x):\n    return x"


def test_extract_code_drops_self_tests_and_demo_calls():
    text = ("```python\nimport math\nLIMIT = 10\ncache = {}\n\ndef solve(x):\n    return x\n\n"
            "assert solve(1) == 2\nprint(solve(3))\nresult = solve(4)\n"
            "if __name__ == '__main__':\n    solve(5)\n```")
    assert extract_code(text, "solve") == (
        "import math\nLIMIT = 10\ncache = {}\n\ndef solve(x):\n    return x")


def test_extract_code_handles_unterminated_fence():
    assert "def solve" in extract_code("```python\ndef solve(x):\n    return x\n", "solve")


def test_parse_assert_cases_understands_common_shapes():
    text = """```python
assert f([1, 2]) == 3
assert 0 == f([])
assert f([5])
assert not f([-1])
assert f([1]) is None
assert other(1) == 2
assert f([1, 2]) == 3
print("not a test")
```"""
    cases = parse_assert_cases(text, "f")
    assert cases == [
        {"call": "f([1, 2])", "expected": "3"},
        {"call": "f([])", "expected": "0"},
        {"call": "f([5])", "expected": "True"},
        {"call": "f([-1])", "expected": "False"},
        {"call": "f([1])", "expected": "None"},
    ]


def test_doc_examples_come_from_the_docstring():
    [task] = load_humaneval(task_ids=["HumanEval/0"])
    cases = doc_example_cases(task)
    assert cases == [
        {"call": "has_close_elements([1.0, 2.0, 3.0], 0.5)", "expected": "False"},
        {"call": "has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)", "expected": "True"},
    ]


def test_grading_uses_hidden_tests():
    [task] = load_humaneval(task_ids=["HumanEval/0"])
    good = ("def has_close_elements(numbers, threshold):\n"
            "    return any(abs(a - b) < threshold for i, a in enumerate(numbers)"
            " for b in numbers[i + 1:])\n")
    assert grade(task, good) is True
    assert grade(task, "def has_close_elements(numbers, threshold):\n    return False\n") is False
    assert grade(task, "") is False


def test_sample_is_seeded_and_sorted():
    a = [t.task_id for t in load_humaneval(n=10, seed=3)]
    assert a == [t.task_id for t in load_humaneval(n=10, seed=3)]
    assert a == sorted(a, key=lambda x: int(x.split("/")[1]))
    assert len(load_humaneval()) == 164


def test_static_scan_flags_risky_code_only():
    assert static_scan("import os\nos.system('rm -rf /')\n") == ["imports os (line 1)"]
    assert static_scan("def f(s):\n    return eval(s)\n") == ["calls eval() (line 2)"]
    assert static_scan("import math\ndef f(x):\n    return math.sqrt(x)\n") == []
