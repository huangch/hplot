import re
from pathlib import Path

P = Path("/workspace/wsinsight/hplot/hplot/tl.py")
lines = P.read_text().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("def gene_layer_matrix_h5ad"))
end = next((i for i in range(start + 1, len(lines))
            if lines[i].startswith("def ") or lines[i].startswith("class ")), len(lines))
print(f"removing tl.py lines {start + 1}..{end} ({end - start} lines)")
print(f"  first: {lines[start].rstrip()!r}")
print(f"  next : {lines[end].rstrip()!r}" if end < len(lines) else "  (to EOF)")
out = lines[:start] + lines[end:]
P.write_text("".join(out).rstrip("\n") + "\n")
print(f"tl.py: {len(lines)} -> {len(out)} lines")

I = Path("/workspace/wsinsight/hplot/hplot/__init__.py")
t = I.read_text()
t = t.replace("    pathway_competitive_test,\n    hpathway_competitive_grid,\n)",
              "    pathway_competitive_test,\n    hpathway_layer_ora,\n)")
t = t.replace("    pathway_layer_profile_h5ad,\n    gene_layer_matrix_h5ad,\n)",
              "    pathway_layer_profile_h5ad,\n)")
t = t.replace('    "pathway_competitive_test",\n    "hpathway_competitive_grid",',
              '    "pathway_competitive_test",\n    "hpathway_layer_ora",')
t = t.replace('    "pathway_layer_profile_h5ad",\n    "gene_layer_matrix_h5ad",',
              '    "pathway_layer_profile_h5ad",')
I.write_text(t)
for bad in ("hpathway_competitive_grid", "gene_layer_matrix_h5ad"):
    assert bad not in t, f"{bad} still referenced in __init__.py"
assert "hpathway_layer_ora" in t
print("__init__.py updated")

S = Path("/workspace/wsinsight/hplot/hplot/stats.py")
s = S.read_text()
s = s.replace("Pair this\n       function with :func:`pathway_competitive_test` before naming any row.",
              "Use\n       :func:`hpathway_layer_ora` (layer-resolved) or\n"
              "       :func:`pathway_competitive_test` (pooled) before naming any row.")
S.write_text(s)
print("stats.py warning now points at hpathway_layer_ora")
print("\nresidual references anywhere in the package:")
for p in Path("/workspace/wsinsight/hplot/hplot").rglob("*.py"):
    for i, l in enumerate(p.read_text().splitlines(), 1):
        if re.search(r"hpathway_competitive_grid|gene_layer_matrix_h5ad|_matched_draws", l):
            print(f"  {p.name}:{i}: {l.strip()[:90]}")
