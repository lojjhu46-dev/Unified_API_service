"""安全计算器"""

import re
import ast
import math
import operator
from app.observability.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_PATTERN = re.compile(r'^[\d\s+\-*/().,%πA-Za-z°℃℉^<>=!]+$')
_TEMP_UNIT_PATTERN = re.compile(r'[°℃℉]')
_TEMP_LABEL_PATTERN = re.compile(r'([\d.]+)\s*[°℃℉]?\s*([CFcf])\b')
_IMPLICIT_MULT = re.compile(r'(\d)\(')
_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 100
_MAX_ABS_RESULT = 1e12

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS = {
    "pi": math.pi,
    "PI": math.pi,
    "π": math.pi,
    "e": math.e,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
}

_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _check_result_size(value):
    if abs(value) > _MAX_ABS_RESULT:
        raise ValueError(f"计算结果过大，已超过安全上限 {_MAX_ABS_RESULT:g}")
    return value


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"不支持的变量或常量: {node.id}")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _check_result_size(_UNARY_OPERATORS[type(node.op)](_eval_ast(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError(f"幂指数过大，最大允许指数为 {_MAX_POWER_EXPONENT}")
        return _check_result_size(_BINARY_OPERATORS[type(node.op)](left, right))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("只允许调用 abs、round、min、max、pow、sqrt 这些数学函数")
        if node.keywords:
            raise ValueError("暂不支持关键字参数，请使用位置参数")
        args = [_eval_ast(arg) for arg in node.args]
        if node.func.id == "pow" and len(args) >= 2 and abs(args[1]) > _MAX_POWER_EXPONENT:
            raise ValueError(f"幂指数过大，最大允许指数为 {_MAX_POWER_EXPONENT}")
        return _check_result_size(_FUNCTIONS[node.func.id](*args))

    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE_OPERATORS:
                raise ValueError("只支持 ==、!=、<、<=、>、>= 这些比较运算")
            right = _eval_ast(comparator)
            if not _COMPARE_OPERATORS[type(op)](left, right):
                return False
            left = right
        return True

    raise ValueError(f"不支持的表达式语法: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_ast(tree)


def _preprocess_expression(raw: str) -> str:
    expr = raw.strip()
    expr = _TEMP_LABEL_PATTERN.sub(r'\1', expr)
    expr = _TEMP_UNIT_PATTERN.sub('', expr)
    expr = expr.replace("^", "**")
    expr = _IMPLICIT_MULT.sub(r'\1*(', expr)
    return expr


def calculate(expression: str) -> dict:
    """执行安全的数学计算"""
    if not expression or not expression.strip():
        return {"success": False, "error": "计算器收到了空的表达式", "result": None}

    processed = _preprocess_expression(expression)

    if len(processed) > _MAX_EXPRESSION_LENGTH:
        return {
            "success": False,
            "error": f"表达式过长（{len(processed)} 个字符），最多支持 {_MAX_EXPRESSION_LENGTH} 个字符",
            "result": None,
        }

    if not _ALLOWED_PATTERN.match(processed):
        illegal_chars = set(c for c in processed if not _ALLOWED_PATTERN.match(c))
        return {
            "success": False,
            "error": f"表达式包含不支持的字符: {''.join(sorted(illegal_chars))}",
            "result": None,
        }

    try:
        result = safe_eval(processed)
        if isinstance(result, float):
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.4f}".rstrip('0').rstrip('.')
        else:
            result_str = str(result)

        return {
            "success": True,
            "result": result_str,
            "expression": expression,
            "error": None,
        }
    except ZeroDivisionError:
        return {"success": False, "error": "数学错误: 除数为零", "result": None}
    except SyntaxError as e:
        return {"success": False, "error": f"表达式语法错误: {e}", "result": None}
    except ValueError as e:
        return {"success": False, "error": f"表达式不安全或不受支持: {e}", "result": None}
    except Exception as e:
        return {"success": False, "error": f"计算出错 ({type(e).__name__}): {e}", "result": None}
