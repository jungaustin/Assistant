"""Tests for the safe calculator (math_tools.calculate).

Covers:
  - basic arithmetic and operator precedence
  - float formatting (trailing zeros trimmed, ints stay ints)
  - whitelisted functions work; unknown names are rejected
  - anything beyond arithmetic (imports, attributes, strings) is rejected
  - division by zero and syntax errors return readable messages, not raises
  - exponentiation bombs are capped
  - MathTools exposes the calculate tool
"""

from __future__ import annotations

from robot.tools.inner.math_tools import MathTools, calculate


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_basic_addition():
    assert calculate("500 + 700") == "500 + 700 = 1200"


def test_precedence_and_parens():
    assert calculate("(500 + 700) / 2") == "(500 + 700) / 2 = 600"
    assert calculate("2 + 3 * 4") == "2 + 3 * 4 = 14"


def test_percentage_style():
    assert calculate("2000 * 0.15") == "2000 * 0.15 = 300"


def test_float_result_trims_trailing_zeros():
    assert calculate("1 / 4") == "1 / 4 = 0.25"
    assert calculate("10 / 3") == "10 / 3 = 3.3333"


def test_negative_numbers():
    assert calculate("-5 + 3") == "-5 + 3 = -2"


def test_whitelisted_functions():
    assert calculate("round(10 / 3)") == "round(10 / 3) = 3"
    assert calculate("max(1, 2, 3)") == "max(1, 2, 3) = 3"
    assert calculate("sqrt(16)") == "sqrt(16) = 4"


def test_modulo_and_floordiv():
    assert calculate("17 % 5") == "17 % 5 = 2"
    assert calculate("17 // 5") == "17 // 5 = 3"


# ---------------------------------------------------------------------------
# rejection / safety
# ---------------------------------------------------------------------------


def test_rejects_names_and_imports():
    assert calculate("__import__('os').system('ls')").startswith("Error")
    assert calculate("x + 1").startswith("Error")


def test_rejects_strings():
    assert calculate("'a' * 9999").startswith("Error")


def test_rejects_attribute_access():
    assert calculate("(1).__class__").startswith("Error")


def test_rejects_unknown_function():
    assert calculate("eval(1)").startswith("Error")


def test_rejects_keyword_arguments():
    assert calculate("round(1.5, ndigits=0)").startswith("Error")


def test_rejects_comparison_and_boolean():
    assert calculate("1 < 2").startswith("Error")
    assert calculate("True + 1").startswith("Error")


def test_power_bomb_is_capped():
    result = calculate("9 ** 9 ** 9")
    assert result.startswith("Error")


def test_too_long_expression():
    assert calculate("1 + " * 100 + "1").startswith("Error")


# ---------------------------------------------------------------------------
# readable errors, not raises
# ---------------------------------------------------------------------------


def test_division_by_zero_message():
    assert calculate("1 / 0") == "Error: division by zero."


def test_syntax_error_message():
    result = calculate("2 +* 3")
    assert result.startswith("Error")
    assert "not a valid arithmetic expression" in result


# ---------------------------------------------------------------------------
# tool wiring
# ---------------------------------------------------------------------------


def test_math_tools_exposes_calculate():
    tool = MathTools().create_calculate_tool()
    assert tool.name == "calculate"
    assert tool.run({"expression": "2 + 2"}) == "2 + 2 = 4"
