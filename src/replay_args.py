"""Print the complete train.py flag list a model was trained with, with overrides, so a rerun is an exact replay.

usage: python3 src/replay_args.py configs/members/RUN.json key=value [key=value ...]
A value of "none" drops the flag; every flag train.py accepts is emitted.
"""
import json
import re
import sys

args = json.load(open(sys.argv[1]))
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    args[k.replace("-", "_")] = None if v == "none" else {"true": True, "false": False}.get(v.lower(), v)

src = open(__file__.replace("replay_args.py", "train.py")).read()
known = {m.replace("-", "_") for m in re.findall(r'add_argument\("--([a-z0-9-]+)"', src)}
unknown = sorted(set(args) - known)
if unknown:
    print(f"replay_args: not train.py flags, skipped: {unknown}", file=sys.stderr)

out = []
for k, v in args.items():
    if k not in known or v is None or v is False or v == "":
        continue
    flag = "--" + k.replace("_", "-")
    if v is True:
        out.append(flag)
    elif isinstance(v, list):
        out += [flag, ",".join(str(x) for x in v)]
    else:
        out += [flag, str(v)]
print(" ".join(out))
