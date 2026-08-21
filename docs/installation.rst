Installation
============

Editable install from the repository:

.. code-block:: bash

   pip install -e .

**Hard dependencies:** ``pandas``, ``numpy``, ``scipy``, ``matplotlib``,
``pygam``.

Optional extras
---------------

The AnnData / scanpy interface (:doc:`anndata`) works out of the box — ``anndata``
is a core dependency. ``squidpy`` is never imported by hplot and is only needed if
you want to build the spatial graph yourself:

.. code-block:: bash

   pip install -e ".[squidpy]"    # adds squidpy (>=1.2)
   pip install -e ".[docs]"       # tooling to build this documentation

Because ``pp`` / ``tl`` / ``pl`` import ``anndata`` lazily (inside the
functions), ``import hplot`` and ``import hplot.core`` stay cheap even though
``anndata`` ships as a core dependency.

Docker
------

A container is provided for paper reproducibility (no local Python setup):

.. code-block:: bash

   docker build -t hplot .
   docker run --rm -v "$PWD":/data hplot test -i /data/data.csv \
       --target immune_fraction --group hpv_status --permutations 999
