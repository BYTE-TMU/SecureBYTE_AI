# task #7

from typing import List, Tuple
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import LLMManager

DEFAULT_RULES = [
    {
        "name": "No eval",
        "pattern": r"\beval\s*\(",
        "message": "Using eval() is dangerous."
    },
    {
        "name": "No exec",
        "pattern": r"\bexec\s*\(",
        "message": "Using exec() is dangerous."
    },
    {
        "name": "No hardcoded passwords",
        "pattern": r"password\s*=\s*['\"].+['\"]",
        "message": "Hardcoded passwords are insecure."
    }
]

def run_hardcoded_checks(code: str, rules: List[dict]) -> List[str]:
    warnings = []
    for rule in rules:
        if re.search(rule["pattern"], code):
            warnings.append(f"{rule['name']}: {rule['message']}")
    return warnings

def run_ai_safety_check(code: str) -> Tuple[bool, str]:
    llm = LLMManager()
    
    prompt = f"""
    You are a security-aware Python code reviewer.
    Analyze the following code for safety, vulnerabilities, or malicious patterns.
    Return ONLY 'SAFE' if it is safe or 'UNSAFE: <reason>' if unsafe.
    Do NOT modify the code.
    Code:
    {code}
    """
    response = llm.generate_response(user_prompt=prompt).strip()
    
    if response.upper().startswith("SAFE"):
        return True, "Code is safe according to AI."
    else:
        return False, response

def run_vulnerability_checks(code: str, rules: List[dict] = DEFAULT_RULES) -> List[str]:
    warnings = run_hardcoded_checks(code, rules)
    
    ai_safe, ai_message = run_ai_safety_check(code)
    if not ai_safe:
        warnings.append(f"AI Check: {ai_message}")
    
    return warnings