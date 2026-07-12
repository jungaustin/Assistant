"""Safe arithmetic calculator tool for Nemo.

Evaluates Python arithmetic expressions by walking the AST and allowing
only numeric literals, arithmetic operators, and a small function
whitelist. This is deliberately NOT eval()/exec(): the expression string
comes from the model, and the model's input comes from voice transcription
of whoever is in the room.

Scope note: this is for quick math on numbers from the conversation
("what's 15% of 2000"). Multi-day log aggregation stays in TrackerDB's
entry_stats — feeding query_entries rows through here reintroduces the
transcription errors entry_stats exists to prevent.
"""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import BaseTool, StructuredTool

_MAX_EXPRESSION_LENGTH = 200


def _safe_pow(base: float, exp: float) -> float:
    # Unbounded ** lets a single expression like 9**9**9 pin the CPU for
    # minutes; these caps cover any real conversational use.
    if abs(exp) > 64 or abs(base) > 1e15:
        raise ValueError("numbers too large for exponentiation")
    return operator.pow(base, exp)


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(_eval_node(arg) for arg in node.args))
        raise ValueError(f"only these functions are allowed: {', '.join(sorted(_FUNCS))}")
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Returns 'expr = result' or an error string.

    Errors are returned (not raised) so the agent loop sees a readable
    message and can rephrase or tell the user, instead of a traceback.
    """
    expression = expression.strip()
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return f"Error: expression too long (max {_MAX_EXPRESSION_LENGTH} characters)."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except SyntaxError:
        return f"Error: {expression!r} is not a valid arithmetic expression."
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, OverflowError) as e:
        return f"Error: {e}."

    result_str = str(int(result)) if float(result) == int(result) else f"{result:.4f}".rstrip("0").rstrip(".")
    return f"{expression} = {result_str}"


class MathTools:
    """LangChain StructuredTool wrapper around the safe calculator."""

    def create_calculate_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description=(
                "Do arithmetic exactly. ALWAYS use this instead of computing "
                "numbers in your head — for percentages, budgets, conversions, "
                "splitting amounts: 'what's 15% of 2000', 'how many calories do "
                "I have left', '72 kg in pounds'.\n\n"
                "Arguments:\n"
                "  expression (str): a Python arithmetic expression using numbers, "
                "+ - * / // % ** and parentheses. Functions allowed: abs, min, "
                "max, round, sqrt. No variables. "
                "Examples: '2000 * 0.15', '(500 + 700) / 2', '72 * 2.20462'.\n\n"
                "Returns the expression with its result, e.g. '2000 * 0.15 = 300'. "
                "Report the result verbatim.\n\n"
                "EXCEPTION: for totals or averages of logged data across days, use "
                "entry_stats — do not copy query_entries rows into this tool."
            ),
        )
