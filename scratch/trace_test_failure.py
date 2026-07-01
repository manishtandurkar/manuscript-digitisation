from pathlib import Path

log_path = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\.system_generated\tasks\task-2140.log")
if log_path.exists():
    text = log_path.read_text(encoding='utf-8', errors='replace')
    # find where single_image_test.py traceback is
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "single_image_test" in line:
            print("\n".join(lines[idx:idx+25]))
            break
else:
    print("Log not found.")
