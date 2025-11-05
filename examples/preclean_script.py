import ast
import astunparse

def clean_code_input(source_code: str) -> str:
    """
    Cleans up Python source code by removing comments, docstrings, and unnecessary whitespace.
    Returns the cleaned version of the code.
    """
    try:
        tree = ast.parse(source_code)

        cleaned_code = astunparse.unparse(tree)

        cleaned_code = "\n".join(line.rstrip() for line in cleaned_code.splitlines() if line.strip())

        return cleaned_code.strip()
    except Exception as e:
        print(f"⚠️ Error while cleaning code: {e}")
        return source_code


if __name__ == "__main__":
    print("🧹 Pre-Clean Code Tool")
    print("Paste your Python code below. Type 'END' on a new line when done.\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    original_code = "\n".join(lines)
    cleaned = clean_code_input(original_code)

    print("\n✅ Cleaned Code:\n")
    print(cleaned)
