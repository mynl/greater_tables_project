.. greater_tables documentation master file, created by
   sphinx-quickstart on Sun Mar  9 08:18:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to greater_tables's documentation!
============================================

|image1| |image2| |image3|

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   homepage
   versions
   greater_tables
   greater_tables.data


**Greater Tables** is a Python tool for producing high-quality, static display tables—intended for use in journal articles, books, formal reports, and printed financial statements. It turns your pandas DataFrame into a clean, black-and-white table—ready for print, PDF, or web. It produces consistent, typographically sound output in HTML, LaTeX (via TikZ), and plain text.

It’s opinionated but flexible, with many options and sensible defaults. Designed for use in **Jupyter Lab**, **Quarto**, and scripting environments, it auto-detects the output format and renders accordingly. Display tables are small and focused—the end result of your analysis, after selecting rows and columns, ordering, and labeling. Greater Tables helps you get those raw materials onto the page, cleanly and consistently.

```python
from greater_tables import GT
GT(df)
```

Or use `display(GT(df))` in notebooks and Quarto documents. Once created, a `GT(df)` object is immutable; re-create it to apply new options. Arguments can be passed directly or loaded from a YAML config file—validated using `pydantic`.

Greater Tables offers similar functionality to `pandas.DataFrame.to_html`, `to_latex`, and `to_markdown`, but with tighter control, better defaults, and no reliance on `pandas` internals. The LaTeX backend uses TikZ for precise control over layout and grid lines.

This is a tool for serious tables—no sparklines, colors, or shading. Just your data, rendered cleanly.

Also included: **Fabricator**, a flexible test DataFrame generator—specify row count, index and column hierarchies, data types, missing values, and more.

Installation
------------

.. code:: python

   pip install greater-tables

Documentation
-------------

`ReadtheDocs <https://greater-tables-project.readthedocs.io/en/latest>`__.

Source
------

`GitHub <https://www.github.com/mynl/greater_tables_project>`__.

Licence
-------

MIT.

Examples
--------


.. code:: python

   import pandas as pd
   import numpy as np
   from greater_tables import sGT
   level_1 = ["Group A", "Group A", "Group B", "Group B", 'Group C']
   level_2 = ['Sub 1', 'Sub 2', 'Sub 2', 'Sub 3', 'Sub 3']

   multi_index = pd.MultiIndex.from_arrays([level_1, level_2])
   start = pd.Timestamp.today().normalize()
   end = pd.Timestamp(f"{start.year}-12-31")  # End of the year
   df = pd.DataFrame(
   {'year': np.arange(2020, 2025, dtype=int),
   'a': np.array((100, 105, 2000, 2025, 100000), dtype=int),
   'b': 10. ** np.linspace(-9, 9, 5),
   'c': np.linspace(601, 4000, 5),
   'd': pd.date_range(start=start, end=end, periods=5),
   'e': 'once upon a time, risk is hard to define, not in Kansas anymore, neutrinos are hard to detect,  $\\int_\\infty^\\infty e^{-x^2/2}dx$ is a hard integral'.split(',')
   }).set_index('year')
   df.columns = multi_index
   gtc.GT(df, caption='A simple GT table.',
          year_cols='year',
          vrule_widths=(1,.5, 0))

.. figure:: img/simple-example.png


The output illustrates:

-  Quarto or Jupyter automatically calls the class’s ``_repr_html_``
   method (or ``_repr_latex_`` for pdf/TeX/Beamer output), providing
   seamless integration across different output formats. ``print()``
   produces fixed-pitch text output.
-  Text is left-aligned, numbers are right-aligned, and dates are
   centered.
-  The index is displayed, and formatted without a comma separator,
   being specified in ``year_cols``. Columns specified in ``ratio_col``
   use % formatting. Explicit control provided over all columns; these
   are just helpers.
-  The first column of integers with a comma thousands separator and no
   decimals.
-  The second column of floats spans several orders of magnitude and is
   formatted using Engineering format, n for nano through k for kilo.
-  The third column of floats is formatted with a comma separator and
   two decimals, based on the average absolute value.
-  The fourth column of date times is formatted as ISO standard dates.
-  Text, in the last column, is sensibly wrapped and can include TeX.
-  The vertical lines separate the levels of the column multiindex.

The Name
--------

Obviously, the name is a play on the ``great_tables`` package. I have
been maintaining a set of macros called
`GREATools <https://www.mynl.com/old/GREAT/home.html>`__ (generalized,
reusable, extensible actuarial tools) in VBA and Python since the late
1990s, and call all my macro packages *GREAT*.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. |image1| image:: https://img.shields.io/github/commit-activity/y/mynl/greater_tables_project
.. |image2| image:: https://img.shields.io/pypi/format/greater_tables
.. |image3| image:: https://img.shields.io/readthedocs/greater_tables_project
.. |image4| image:: docs/img/simple-example.png
