# `greater_tables` Project

![](https://img.shields.io/github/commit-activity/y/mynl/greater_tables_project)
![](https://img.shields.io/pypi/format/greater_tables)
![](https://img.shields.io/readthedocs/greater_tables_project)


## Greater Tables

Creating presentation quality tables is difficult. `greater_tables` provides
a flexible way to create consistent tables in HTML, LaTeX (PDF), and terminal
text outputs from Pandas dataframes. 
It has many options but sensible defaults. It is designed
for use in Jupyter Lab and Quarto and will seamlessly return the correct format
for each output type. The basic usage is simply:

```python
from greater_tables import GT 
# ...create dataframe df...
GT(df)
```

or `display(GT(df))` if called within a Jupyter or Quarto code block. Once created `GT(df)` is immutable; to change options re-create. Presentation tables are small! They fit on one or two pages and, while `GT` does a lot of work to determine formating options, it still runs very quickly. 

`greater_tables`  provides similar functionality to pandas `to_html`, `to_latex` and `to_markdown` 
methods, without relying on them, and improves them in various ways. LaTeX output uses Tikz tables for very tight control over layout and grid lines.  Arguments can be passed directly or set via a YAML configuration file. Validation is handled by `pydantic`.

The package is tailored to more austere, black-and-white tables: no sparklines, colors or background shading. Tables can include a simple caption, but not more elaborate headers and footers. 


## Installation

```python
pip install greater-tables
```

## Documentation

[ReadtheDocs](https://greater-tables-project.readthedocs.io/en/latest).

## Source

[GitHub](https://www.github.com/mynl/greater_tables_project).

## Licence

MIT.

## Examples

```python
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
```

![](docs/img/simple-example.png)

The output illustrates:

-   Quarto or Jupyter automatically calls the class's `_repr_html_` method (or
    `_repr_latex_` for pdf/TeX/Beamer output), providing seamless
    integration across different output formats. `print()` produces fixed-pitch text output. 
-   Text is left-aligned, numbers are right-aligned, and dates are centered.
-   The index is displayed, and formatted without a comma separator, being specified in `year_cols`. Columns specified in `ratio_col` use % formatting. Explicit control provided over all columns; these are just helpers. 
-   The first column of integers with a comma thousands separator and no decimals.
-   The second column of floats spans several orders of magnitude and is
    formatted using Engineering format, n for nano through k for kilo.
-   The third column of floats is formatted with a comma separator and
    two decimals, based on the average absolute value.
-   The fourth column of date times is formatted as ISO standard dates.
- Text, in the last column, is sensibly wrapped and can include TeX. 
-   The vertical lines separate the levels of the column multiindex.



## The Name

Obviously, the name is a play on the `great_tables` package. I have
been maintaining a set of macros called
[GREATools](https://www.mynl.com/old/GREAT/home.html) (generalized,
reusable, extensible actuarial tools) in VBA and Python since the late
1990s, and call all my macro packages *GREAT*.


## History

3.3.0
-------
* Added `tikz_` series of options to config: column and row separation,  
  container_env (for e.g., sidewaystable), hrule and vrule indices.

3.2.0
-------
* Added more tex snippets!
* Refactored tikz and column width behavior

3.1.0
-------
* adjustments for auto format
* rearranged gtcore order of methods

3.0.0
-------

* config files / pydantic config input 
* unified col width and info dataframe
* de-texing
* cli for config and writeout a csv etc.

* testdf suite
* Automated TeX to SVG 

2.0.0
------

* **v2.0.0** solid release old-style, all-argument GT
* Better column widths
* Custom text output 
* Rich table output 

1.1.1
-------
* Added logo, updated docs.

1.1.0
------

* added ``formatters`` argument to pass in column specific formatters by name as a number (``n`` converts to ``{x:.nf}``, format string, or function
* Added ```tabs`` argument to provide column widths
* Added ``equal`` argument to provide hint that column widths should all be equal
* Added ``caption_align='center'`` argument to set the caption alignment
* Added ``large_ok=False`` argument, if ``False`` providing a dataframe with more than 100 rows throws an error. This function is expensive and is designed for small frames.

1.0.0
------

* Allow input via list of lists, or markdown table
* Specify overall float format for whole table
* Specify column alingment with 'llrc' style string
* ``show_index`` option
* Added more tests
* Docs updated
* Set tabs for width; use of width in HTML format.

0.6.0
------

* Initial release

Early development
-------------------

* 0.1.0 - 0.5.0: Early development
* tikz code from great.pres_manager


## 📁 Project Layout

```
greater_tables_project/
|   LICENSE
|   pyproject.toml
|   README.md
|   
+---dist
|       
+---docs
|   |   books.bib
|   |   conf.py
|   |   greater_tables.data.rst
|   |   greater_tables.rst
|   |   index.rst
|   |   library.bib
|   |   make.bat
|   |   Makefile
|   |   modules.rst
|   |   start-server.bat
|   |   style.csl
|   |   
+---greater_tables
|   |   __init__.py
|   |   cli.py
|   |   gtconfig.py
|   |   gtcore.py
|   |   gtenums.py
|   |   gtformats.py
|   |   hasher.py
|   |   testdf.py
|   |   tex_svg.py
|   |   
|   +---data
|   |   |   __init__.py
|   |   |   tex_list.csv
|   |   |   tex_list.py
|   |   |   words-12.md
|           
+---greater_tables.egg-info
|       
+---img
|       hard-html.png
|       hard-tex.png
```
