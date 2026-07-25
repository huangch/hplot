"""Pathway/gene-set catalogs for the H-Pathway Summary, cached locally.

Three sources, all resolvable to a local ``.gmt`` under ``cache_dir``:

- ``"msigdb"``   : MSigDB Hallmark (via Enrichr ``MSigDB_Hallmark_2020``).
- ``"go_bp"``    : GO Biological Process (via Enrichr ``GO_Biological_Process_2021``).
- ``"go_goatools"`` : GO built locally from a ``go-basic.obo`` + NCBI ``gene2go``
  (human subset), with true DAG propagation to ancestor terms, mapped to gene
  symbols via NCBI ``Homo_sapiens.gene_info``. This is the default.

Each catalog is materialised once as a ``.gmt`` and reused. :func:`load_catalog`
returns ``{set_name: [gene_symbol, ...]}``. :func:`select_signatures_on_panel`
then gates those sets against the genes actually present on an assay panel.

``goatools`` is imported lazily, only when ``source="go_goatools"`` is built for
the first time, so the rest of :mod:`hplot` has no hard dependency on it.
"""
from __future__ import annotations

import gzip
import os
import ssl
import urllib.request

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

_ENRICHR = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}"
_GENE_INFO_URL = ("https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
                  "Homo_sapiens.gene_info.gz")
_ENRICHR_LIB = {"msigdb": "MSigDB_Hallmark_2020", "go_bp": "GO_Biological_Process_2021"}


# --------------------------------------------------------------------------- #
# GMT read / write
# --------------------------------------------------------------------------- #
def read_gmt(path):
    sets = {}
    with open(path, "r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            if genes:
                sets[name] = sorted(set(genes))
    return sets


def write_gmt(path, sets):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for name, genes in sets.items():
            fh.write("\t".join([name, ""] + list(genes)) + "\n")


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180, context=_SSL) as r, open(dest, "wb") as w:
        w.write(r.read())
    return dest


# --------------------------------------------------------------------------- #
# Enrichr-based catalogs (MSigDB Hallmark, GO-BP)
# --------------------------------------------------------------------------- #
def _build_enrichr_gmt(source, gmt_path):
    lib = _ENRICHR_LIB[source]
    req = urllib.request.Request(_ENRICHR.format(lib=lib),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180, context=_SSL) as r:
        text = r.read().decode("utf-8", "replace")
    sets = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        # Enrichr genes may carry a ",weight" suffix; strip it and upper-case.
        genes = [g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()]
        genes = [g for g in genes if g]
        if genes:
            sets[parts[0]] = sorted(set(genes))
    write_gmt(gmt_path, sets)
    return sets


# --------------------------------------------------------------------------- #
# goatools-based GO catalog (local OBO + gene2go + gene_info)
# --------------------------------------------------------------------------- #
def _gene_id_to_symbol(gene_info_gz):
    """GeneID(int) -> Symbol from NCBI Homo_sapiens.gene_info(.gz)."""
    gid2sym = {}
    op = gzip.open if str(gene_info_gz).endswith(".gz") else open
    with op(gene_info_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            try:
                gid2sym[int(f[1])] = f[2]
            except ValueError:
                continue
    return gid2sym


def _filter_human_gene2go(gene2go_path, out_path, taxid="9606"):
    """Stream the (possibly multi-GB, all-species) gene2go to a human subset."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    op = gzip.open if str(gene2go_path).endswith(".gz") else open
    n = 0
    with op(gene2go_path, "rt") as fh, open(out_path, "w") as w:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.startswith(taxid + "\t"):
                w.write(line)
                n += 1
    return out_path


def _build_go_goatools_gmt(gmt_path, *, obo_path, gene2go_path, cache_dir,
                           namespaces=("BP", "MF", "CC"), min_genes=3,
                           propagate=True, progress=True):
    from goatools.obo_parser import GODag

    gene_info_gz = _download(_GENE_INFO_URL, os.path.join(cache_dir, "Homo_sapiens.gene_info.gz"))
    gid2sym = _gene_id_to_symbol(gene_info_gz)
    human_g2g = _filter_human_gene2go(gene2go_path, os.path.join(cache_dir, "human_gene2go.tsv"))

    godag = GODag(obo_path, prt=None)
    _NS = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}
    want_ns = set(namespaces)

    # direct GO -> set(GeneID), skipping NOT-qualified annotations
    go2genes = {}
    with open(human_g2g, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            geneid, go_id, qualifier = f[1], f[2], f[4]
            if "NOT" in qualifier:
                continue
            try:
                gid = int(geneid)
            except ValueError:
                continue
            go2genes.setdefault(go_id, set()).add(gid)

    # DAG propagation: push each term's genes up to all ancestor terms
    prop = {}
    for go_id, genes in go2genes.items():
        term = godag.get(go_id)
        if term is None:
            continue
        targets = {go_id}
        if propagate:
            targets |= set(term.get_all_parents())
        for t in targets:
            prop.setdefault(t, set()).update(genes)

    sets = {}
    for go_id, gids in prop.items():
        term = godag.get(go_id)
        if term is None:
            continue
        ns = _NS.get(term.namespace, "??")
        if ns not in want_ns:
            continue
        syms = sorted({gid2sym[g] for g in gids if g in gid2sym})
        if len(syms) < min_genes:
            continue
        name = f"{term.name} ({go_id}|{ns})"
        sets[name] = syms

    write_gmt(gmt_path, sets)
    return sets


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def load_catalog(source, *, cache_dir, obo_path=None, gene2go_path=None,
                 force=False, **kwargs):
    """Return ``{set_name: [symbols]}`` for the requested catalog, cached as GMT.

    Parameters
    ----------
    source : {"msigdb", "go_bp", "go_goatools"}
        Catalog to materialise. ``"msigdb"`` / ``"go_bp"`` are fetched from
        Enrichr; ``"go_goatools"`` is built locally from ``obo_path`` +
        ``gene2go_path`` (requires the optional ``goatools`` dependency).
    cache_dir : str
        Directory the ``<source>.gmt`` (and any download side-files) live in.
    obo_path, gene2go_path : str | None
        Required only for ``source="go_goatools"``.
    force : bool
        Rebuild even if a cached ``.gmt`` already exists.
    """
    os.makedirs(cache_dir, exist_ok=True)
    gmt = os.path.join(cache_dir, f"{source}.gmt")
    if os.path.exists(gmt) and os.path.getsize(gmt) > 0 and not force:
        return read_gmt(gmt)
    if source in _ENRICHR_LIB:
        return _build_enrichr_gmt(source, gmt)
    if source == "go_goatools":
        if not (obo_path and gene2go_path):
            raise ValueError("go_goatools requires obo_path and gene2go_path.")
        return _build_go_goatools_gmt(gmt, obo_path=obo_path, gene2go_path=gene2go_path,
                                      cache_dir=cache_dir, **kwargs)
    raise ValueError(f"unknown catalog source: {source!r}")


def select_signatures_on_panel(catalog, panel, *, mode="discovery",
                               min_panel_genes=5, max_panel_genes=250,
                               min_coverage=0.5, min_genes=3):
    """Gate gene-set signatures against the genes present on an assay panel.

    Parameters
    ----------
    catalog : dict[str, sequence[str]]
        ``{signature_name: [genes]}`` candidate universe (e.g. from
        :func:`load_catalog` or a curated user list).
    panel : iterable[str]
        Gene symbols measured by the assay (e.g. ``adata.var_names``).
    mode : {"discovery", "user"}
        ``"discovery"`` keeps sets whose on-panel gene count is within
        ``[min_panel_genes, max_panel_genes]`` (drops tiny and root-like huge
        sets). ``"user"`` keeps a curated list by fractional ``min_coverage``
        and an absolute ``min_genes`` floor.
    min_panel_genes, max_panel_genes : int
        Discovery-mode on-panel gene-count window.
    min_coverage : float
        User-mode minimum fraction of a set's genes that must be on the panel.
    min_genes : int
        User-mode minimum absolute on-panel gene count.

    Returns
    -------
    (present, coverage) : tuple[dict, dict]
        ``present`` maps each kept signature to its on-panel gene list.
        ``coverage`` maps every input signature to
        ``(n_present, n_total, fraction)``.
    """
    if mode not in ("discovery", "user"):
        raise ValueError(f"mode must be 'discovery' or 'user', got {mode!r}.")
    panel = set(map(str, panel))
    present, coverage = {}, {}
    for name, genes in catalog.items():
        uniq = sorted({str(g) for g in genes})
        on_panel = [g for g in uniq if g in panel]
        frac = len(on_panel) / max(len(uniq), 1)
        coverage[name] = (len(on_panel), len(uniq), frac)
        if mode == "discovery":
            if min_panel_genes <= len(on_panel) <= max_panel_genes:
                present[name] = on_panel
        else:
            if frac >= min_coverage and len(on_panel) >= min_genes:
                present[name] = on_panel
    return present, coverage
