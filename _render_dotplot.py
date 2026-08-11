import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from hplot.plotting import plot_hpathway_dotplot

rng = np.random.default_rng(0)
paths = [f"HALLMARK_SET_{i}" for i in range(7)]
layers = list(range(-7, 22))
rows = []
for p in paths:
    for L in layers:
        rows.append({"pathway": p, "layer": L,
                     "score": float(rng.random()),
                     "fdr": float(10 ** -rng.uniform(0, 3.5)),
                     "direction": float(rng.choice([-1.0, 0.0, 1.0]))})
df = pd.DataFrame(rows)


def probe(tag, **kw):
    fig, ax, ax_cbar = None, None, None
    out = plot_hpathway_dotplot(
        df, path_col="pathway", layer_col="layer", score_col="score",
        fdr_col="fdr", direction_col="direction", layer_limits=(-7, 21), **kw)
    if isinstance(out, dict):
        print(f"[{tag}] returned keys: {sorted(out)}")
        fig = next(v for v in out.values()
                   if isinstance(v, matplotlib.figure.Figure)
                   or hasattr(v, "figure"))
        fig = fig if isinstance(fig, matplotlib.figure.Figure) else fig.figure
    else:
        fig = out[0] if isinstance(out, tuple) else out.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    titles = [t for t in fig.findobj(matplotlib.text.Text)
              if t.get_text() in ("elevated vs own-window baseline",
                                  "depressed vs own-window baseline")]
    boxes = sorted((t.get_window_extent(r) for t in titles), key=lambda b: b.x0)
    print(f"[{tag}] figsize={tuple(round(v, 2) for v in fig.get_size_inches())}")
    print(f"[{tag}] ramp titles found: {len(titles)}")
    if len(boxes) == 2:
        gap = boxes[1].x0 - boxes[0].x1
        print(f"[{tag}] horizontal gap between the two titles: {gap:.1f} px "
              f"-> {'OK (no overlap)' if gap > 0 else 'OVERLAP'}")
    legs = [a for a in fig.findobj(matplotlib.legend.Legend)]
    print(f"[{tag}] right-hand legends: {len(legs)} "
          f"({[l.get_title().get_text() for l in legs]})")
    matplotlib.pyplot.close(fig)


probe("long labels / bottom key",
      direction_labels=("depressed vs own-window baseline",
                        "elevated vs own-window baseline"),
      side_colorbar=True)
probe("short labels / bottom key", side_colorbar=True)
probe("no bottom key", side_colorbar=False)
