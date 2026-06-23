from pathlib import Path
log_path = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\.system_generated\tasks\task-2084.log")
if log_path.exists():
    print(log_path.read_text(encoding='utf-8', errors='replace'))
else:
    print("Log file does not exist yet.")
