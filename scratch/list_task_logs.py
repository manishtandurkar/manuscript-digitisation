import os
from pathlib import Path

p = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\.system_generated\tasks")
if p.exists():
    print(f"Task log folder contains: {os.listdir(str(p))}")
else:
    print("Task log folder does not exist")
