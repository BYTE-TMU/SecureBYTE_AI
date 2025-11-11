import os
import json
import pytest

def get_file_line_count(filepath):
    """Return number of lines in file"""""
    with open(filepath, "r", encoding="utf-8") as file :
        contents = file.read()
        line_count = contents.count("\n")
        return line_count
    
def test_model_vulnerability():
    with open("model_output.json", "r") as file:
        data = json.load(file)

    for v in data.get("vulnerabilities", []):
        file_path = v["file"]
        line = v["line"]
        #check file exists
        assert os.path.exists(file_path), f"Hallucination: File does not exist -> {file_path}"

        #check line number is valid
        total_lines = get_file_line_count(file_path)
        assert (line >= 1) and(line <= total_lines), (
            f"Hallucincation: {file_path} has only {total_lines} lines," 
            f"but model referenced line {line}."
        )

