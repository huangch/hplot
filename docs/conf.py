# Configuration file for the Sphinx documentation builder.
#
# Build locally with:
#     pip install -e ".[docs]"
#     sphinx-build -b html docs docs/_build/html

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "hplot"
author = "Chi-Hong Huang et al."
copyright = f"{date.today().year}, {author}"

try:
    from hplot import __version__ as release
except Exception:  # pragma: no cover
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Import-time optional deps that need not be installed to build the docs.
autodoc_mock_imports = ["pygam", "squidpy"]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/stable/", None),
}
# Do not fail the build when inventories cannot be fetched (offline builds).
intersphinx_timeout = 5

# -- HTML output -------------------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Graph-geodesic distance-stratified spatial profiling "
                   "at tissue boundaries",
    "fixed_sidebar": True,
    "page_width": "1000px",
}
