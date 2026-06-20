lines = open("src/binarise.py", encoding="utf-8").readlines()
target = None
for i, l in enumerate(lines):
    if "if doc_type ==" in l and "palm_leaf" in l:
        target = i
        break

if target is None:
    print("NOT FOUND")
else:
    new_block = [
        "    if detect_rubbing(img):\n",
        "        binary = binarise_rubbing(img)\n",
        "    elif doc_type == \"palm_leaf\":\n",
        "        binary = binarise_palm_leaf(img)\n",
        "    else:\n",
        "        binary = binarise_stone(img)\n",
    ]
    before = lines[:target]
    after = lines[target+5:]
    result = before + new_block + after
    open("src/binarise.py", "w", encoding="utf-8").writelines(result)
    print("Replaced block at line", target+1)
