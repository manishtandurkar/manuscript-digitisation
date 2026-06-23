import glob
import os

for root, dirs, files in os.walk("."):
    for f in files:
        if "026" in f:
            full_path = os.path.join(root, f)
            print(f"File: {full_path} | Size: {os.path.getsize(full_path)} bytes")
