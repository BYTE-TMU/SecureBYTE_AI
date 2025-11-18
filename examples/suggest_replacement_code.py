# task #17

import os
import sys
from dotenv import load_dotenv
from preclean_script import preclean_code
from custom_rules import run_vulnerability_checks, DEFAULT_RULES


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import LLMManager

def suggest_replacement_code(bad_code: str) -> str:
    llm = LLMManager()

    bad_code = preclean_code(bad_code)

    prompt = f"""
    You are a secure code analysis and improvement assistant.

    Your tasks:
    1. Analyze the provided Python code for vulnerabilities, unsafe patterns, and syntax issues.
    2. Report any warnings clearly under a "⚠️ Warnings" section.
    3. If the issues are *minor* (e.g., syntax errors, undefined variables, logic mistakes), still attempt to fix and improve the code safely.
    4. Only refuse to improve if the code contains *critical security risks* (like eval(), exec(), system(), subprocess calls, file deletion, or network access).

    Then, output:
    💡 Suggested Replacement:
    {bad_code}
    """
    response = llm.generate_response(user_prompt=prompt)

    cleaned = response.replace("```python", "").replace("```", "").strip()
    return cleaned

def main():
    print("🧩 Suggest Replacement Code Tool")
    print("Paste your Python code. Type 'END' on a new line when finished.\n")

    while True:
        print("Type/copy and paste code below (Type 'END' on a new line when finished):\n")
        lines = []
        while True:
            line = input()
            if line.strip().lower() in ["exit", "quit", "leave"]:
                print("👋 Goodbye!")
                return
            if line.strip().upper() == "END":
                break
            lines.append(line)
        
        if not lines:
            continue
        
        bad_code = "\n".join(lines)

        warnings = run_vulnerability_checks(bad_code, DEFAULT_RULES)
        if warnings:
            print("\n⚠️ Warnings detected in your code:")
            for severity, message in warnings:
                icon = "🚨" if severity == "critical" else "⚠️"
                print(f"{icon} [{severity.upper()}] {message}")

            if any(severity == "critical" for severity, _ in warnings):
                print("\n🚫 Skipping improvement due to critical vulnerabilities.\n")
                print("=" * 60)
                continue
            else:
                user_choice = input("\nProceed with improvement despite warnings? (y/n): ").strip().lower()
                if user_choice != "y":
                    print("🛑 Skipping by user choice.")
                    print("=" * 60)
                    continue

        print("\n💡 Suggested Replacement:\n")
        improved = suggest_replacement_code(bad_code)
        print(improved)
        print("=" * 60)

if __name__ == "__main__":
    main()