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

def run_hardcoded_checks(code: str, rules: List[dict]) -> List[Tuple[str, str]]:
    """Return a list of (severity, message) pairs."""
    warnings = []
    for rule in rules:
        if re.search(rule["pattern"], code):
            severity = "critical" if rule["name"].lower() in ["no eval", "no exec"] else "warning"
            warnings.append((severity, f"{rule['name']}: {rule['message']}"))
    return warnings


def run_vulnerability_checks(code: str, rules: List[dict] = DEFAULT_RULES) -> List[Tuple[str, str]]:
    warnings = run_hardcoded_checks(code, rules)
    
    ai_safe, ai_message = run_ai_safety_check(code)
    if not ai_safe:
        critical_keywords = ["system", "delete", "shutdown", "exec", "eval", "dangerous"]
        severity = "critical" if any(word in ai_message.lower() for word in critical_keywords) else "warning"
        warnings.append((severity, f"AI Check: {ai_message}"))
    
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