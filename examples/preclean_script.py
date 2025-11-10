# task #15

import ast

def preclean_code(source_code: str) -> str | None:
    try:
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0]
                    if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                        node.body.pop(0)

        cleaned_code = ast.unparse(tree)
        return cleaned_code.strip()

    except SyntaxError:
        print("⚠️ Warning: Could not parse input as Python code. Skipping AST cleaning.")
        return None