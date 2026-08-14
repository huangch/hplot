import glob
import os
import subprocess
import sys

os.chdir("/workspace/wsinsight/hplot")
PY = "/opt/anaconda3/envs/wsinsight/bin/python"
fails = []
for f in sorted(glob.glob("test/test_*.py")):
    r = subprocess.run([PY, f], capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    status = "OK  " if r.returncode == 0 else "FAIL"
    last = next((l for l in reversed(tail) if l.strip()), "")
    print(f"  {status}  {f:44s} {last[:70]}")
    if r.returncode != 0:
        fails.append((f, "\n".join(tail[-25:])))

print(f"\n{len(fails)} failing file(s)")
for f, out in fails:
    print(f"\n=== {f} ===\n{out}")
sys.exit(1 if fails else 0)
