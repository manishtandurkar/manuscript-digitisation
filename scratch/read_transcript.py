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
    
    # Filter only steps that are USER_INPUT or PLANNER_RESPONSE (representing high level conversational exchange)
    conversational_steps = []
    for step in steps:
        if step.get("type") in ("USER_INPUT", "PLANNER_RESPONSE"):
            conversational_steps.append(step)
            
    print(f"Total conversational steps: {len(conversational_steps)}")
    for step in conversational_steps[-25:]:
        source = step.get("source")
        content = step.get("content", "")
        # truncate
        lines = content.split('\n')
        trunc = '\n'.join(lines[:6])
        if len(lines) > 6:
            trunc += "\n..."
        print(f"\n[{source}]:\n{trunc}")
