Installation
============

Editable install from the repository:

.. code-block:: bash

   pip install -e .

**Hard dependencies:** ``pandas``, ``numpy``, ``scipy``, ``matplotlib``,
``pygam``.

Optional extras
---------------

The AnnData / scanpy / squidpy interface (:doc:`anndata`) needs extra packages.
The core engine and CLI work without them:

.. code-block:: bash

   pip install -e ".[anndata]"    # adds anndata (>=0.8)
   pip install -e ".[squidpy]"    # adds anndata + squidpy (>=1.2)
   pip install -e ".[docs]"       # tooling to build this documentation

Because ``pp`` / ``tl`` / ``pl`` / ``gr`` import ``anndata`` lazily (inside the
functions), ``import hplot`` and ``import hplot.core`` still work with only the
hard dependencies installed. You only need ``hplot[anndata]`` when you actually
call that interface.

Docker
------

A container is provided for paper reproducibility (no local Python setup):

.. code-block:: bash

   docker build -t hplot .
   docker run --rm -v "$PWD":/data hplot test -i /data/data.csv \
       --target immune_fraction --group hpv_status --permutations 999
