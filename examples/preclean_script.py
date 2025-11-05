import ast

def preclean_code(source_code: str) -> str:
    """
    Remove comments and docstrings from Python code.
    Returns the cleaned code as a string.
    """
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr):
                expr = node.body[0]
                # Remove docstring if first expression is a string literal
                if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                    node.body.pop(0)

    # Convert AST back to code
    cleaned_code = ast.unparse(tree)
    return cleaned_code