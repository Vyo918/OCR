"""The LLM writes arithmetic as {{7.9*3.525*3.2}}; this module evaluates it exactly.

A 4B model reads numbers well but multiplies badly (it wrote 7.9 x 3.52 x 1.85 = 50.4; the product is 51.44), and a volume that is
off by two m3 is a wrong answer. So the model never does the arithmetic: it states the expression, the system substitutes the value.
"""
from __future__ import annotations

import ast
import operator as op
import re

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
# an optional trailing '= 50.9' is the model's own (wrong) result written after the placeholder: it is dropped, the exact value replaces it
_CALC = re.compile(r"\{\{([^{}]+)\}\}(?:\s*[=≈≒]\s*-?\d[\d,]*(?:\.\d+)?)*")


def evaluate(expr: str) -> float:
    """+ - * / ** and parentheses over plain numbers; x / X / × / ÷ / ^ are accepted as operators. Anything else raises ValueError."""
    e = expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    e = re.sub(r"(?<=[\d.)])\s*[xX]\s*(?=[\d.(])", "*", e)

    def go(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -go(n.operand) if isinstance(n.op, ast.USub) else go(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = go(n.left), go(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 6:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        raise ValueError(f"unsupported expression: {expr!r}")

    try:
        return go(ast.parse(e.strip(), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as err:
        raise ValueError(str(err)) from err


def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fill(text: str) -> str:
    """Replace every {{expression}} by 'expression = value' (an expression that cannot be evaluated is left as written)."""
    def one(m: re.Match) -> str:
        try:
            return f"{m.group(1).strip()} = {fmt(evaluate(m.group(1)))}"
        except ValueError:
            return m.group(1).strip()
    return _CALC.sub(one, text)
