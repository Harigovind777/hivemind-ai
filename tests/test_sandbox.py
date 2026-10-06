from hivemind.sandbox import run_cases, run_program

CODE = "def f(x):\n    print('noise')\n    if x < 0:\n        raise ValueError('neg')\n    return x * 1.5\n"


def test_cases_report_outputs_errors_and_pass_status():
    results = run_cases(CODE, [
        {"call": "f(2)", "expected": "3.0"},
        {"call": "f(2)", "expected": "4"},
        {"call": "f(-1)", "expected": "0"},
        {"call": "f(1)"},
    ])
    assert [r.passed for r in results] == [True, False, False, None]
    assert results[0].output == "3.0"
    assert results[2].error == "ValueError"
    assert results[2].signature == "<ValueError>"


def test_float_noise_is_normalised():
    [r] = run_cases("def g():\n    return 0.1 + 0.2\n", [{"call": "g()", "expected": "0.3"}])
    assert r.passed is True


def test_infinite_loop_times_out_without_hanging():
    results = run_cases("def h():\n    while True:\n        pass\n",
                        [{"call": "h()"}, {"call": "1 + 1"}], case_timeout=0.5)
    assert results[0].error == "Timeout"
    assert results[1].output == "2"


def test_code_that_fails_to_load_marks_every_case():
    results = run_cases("def broken(:\n", [{"call": "broken()"}, {"call": "broken()"}])
    assert all(r.error.startswith("load: SyntaxError") for r in results)
    assert not any(r.is_answer for r in results)
    [raised] = run_cases(CODE, [{"call": "f(-1)"}])
    assert raised.is_answer  # raising can be the specified behaviour


def test_run_program_reports_success_and_failure():
    assert run_program("assert 1 + 1 == 2\n")[0] is True
    ok, err = run_program("assert 1 + 1 == 3\n")
    assert ok is False and "AssertionError" in err
