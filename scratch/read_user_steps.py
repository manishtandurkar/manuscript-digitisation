import json
from pathlib import Path

transcript_path = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\.system_generated\logs\transcript.jsonl")

if not transcript_path.exists():
    print("Transcript does not exist")
else:
    steps = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            steps.append(json.loads(line))
            
    print(f"Total steps: {len(steps)}")
    
    types = set()
    sources = set()
    for s in steps:
        types.add(s.get("type", ""))
        sources.add(s.get("source", ""))
    print("Types:", types)
    print("Sources:", sources)
    
    # Let's print the last 15 USER steps (where source starts with USER or type contains USER or source != MODEL)
    user_steps = []
    for s in steps:
        if "USER" in str(s.get("source")) or "USER" in str(s.get("type")):
            user_steps.append(s)
    print(f"Total user steps: {len(user_steps)}")
    for s in user_steps[-10:]:
        print(f"\n--- Step {s.get('step_index')} | Source: {s.get('source')} | Type: {s.get('type')} ---")
        print(s.get("content", ""))
