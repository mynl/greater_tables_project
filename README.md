---
jupyter:
  jupytext:
    formats: ipynb,qmd
    text_representation:
      extension: .qmd
      format_name: quarto
      format_version: 1.0
      jupytext_version: 1.16.4
  keep-ipynb: true
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
toc-title: Table of contents
---

# Greater Tables

Creating presentation quality tables from pandas dataframes is
frustrating. It is hard to left-align text and right-align numbers using
pandas `display` or `df.to_html`. The `great_tables` package does a
really nice job with pandas and polars dataframes but does not support
indexes or TeX output.

This package is provides consistent HTML and TeX table output with
flexible type-based formatting, and table rules. Its main class `GT` can
be subclassed to set appropriate defaults for your project.

The project is currently in **beta** status.

## The Name

Obviously, the name is a play on the `great_tables` package. But, I have
been maintaining a set of macros called
[GREATools](https://www.mynl.com/old/GREAT/home.html) (generalized,
reusable, extensible actuarial tools) in VBA and Python since the late
1990s, and call all my macro packages "GREAT".

## Installation

``` python
pip install greater-tables
```

## Examples

The following example shows quite a hard table. It is formatted using
the `sGT` class, which is a subclass of `GT` with a few defaults set.

``` {.python .cell-code}
import pandas as pd
import numpy as np
from greater_tables import sGT
level_1 = ["Group A", "Group A", "Group B", "Group B", 'Group C']
level_2 = ['Sub 1', 'Sub 2', 'Sub 2', 'Sub 1', 'Sub 1']

multi_index = pd.MultiIndex.from_arrays([level_1, level_2])

start = pd.Timestamp.today().normalize()  # Today's date, normalized to midnight
end = pd.Timestamp(f"{start.year}-12-31")  # End of the year

hard = pd.DataFrame(
{'x': np.arange(2020, 2025, dtype=int), 
'a': np.array((100, 105, 2000, 2025, 100000), dtype=int),
'b': 10. ** np.linspace(-9, 9, 5),
'c': np.linspace(601, 4000, 5),
'd': pd.date_range(start=start, end=end, periods=5),
'e': 'once upon a time, lived happily ever after, not in Kansas anymore, neutrinos are hard to detect, risk is hard to define'.split(',')
}).set_index('x')
hard.columns = multi_index
print(sGT(hard, 'A hard table.').html) 
```

 

 

<style>#TS76FPUXL623T { border-collapse: collapse; font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif; font-size: 0.9em; width: auto; border: none; overflow: auto; } #TS76FPUXL623T caption { padding: 8px 10px 4px 10px; font-size: 0.99em; text-align: center; font-weight: normal; caption-side: top; } #TS76FPUXL623T thead { border-top: 1px solid #000; border-bottom: 1px solid #000; font-size: 0.99em; } #TS76FPUXL623T tbody { border-bottom: 1px solid #000; } #TS76FPUXL623T th { vertical-align: bottom; padding: 8px 10px 8px 10px; } #TS76FPUXL623T td { padding: 4px 10px 4px 10px; vertical-align: top; } #TS76FPUXL623T .grt-hrule-0 { border-top: 0px solid #000; } #TS76FPUXL623T .grt-hrule-1 { border-top: 0px solid #000; } #TS76FPUXL623T .grt-hrule-2 { border-top: 0px solid #000; } #TS76FPUXL623T .grt-bhrule-0 { border-bottom: 1.5px solid #000; } #TS76FPUXL623T .grt-bhrule-1 { border-bottom: 1px solid #000; } #TS76FPUXL623T .grt-vrule-index { border-left: 1.5px solid #000; } #TS76FPUXL623T .grt-vrule-0 { border-left: 1.5px solid #000; } #TS76FPUXL623T .grt-vrule-1 { border-left: 1px solid #000; } #TS76FPUXL623T .grt-vrule-2 { border-left: 0.5px solid #000; } #TS76FPUXL623T .grt-left { text-align: left; } #TS76FPUXL623T .grt-center { text-align: center; } #TS76FPUXL623T .grt-right { text-align: right; font-variant-numeric: tabular-nums; } #TS76FPUXL623T .grt-head { font-family: "Times New Roman", 'Courier New'; font-size: 0.99em; } #TS76FPUXL623T .grt-bold { font-weight: bold; }</style>
<table id="TS76FPUXL623T">
<caption>
A hard table.
</caption>
<thead>
<tr>
<th class="grt-left">
</th>
<th class="grt-center grt-bhrule-0 grt-vrule-index" colspan="2">
Group A
</th>
<th class="grt-center grt-bhrule-0 grt-vrule-0" colspan="2">
Group B
</th>
<th class="grt-center grt-bhrule-0 grt-vrule-0" colspan="1">
Group C
</th>
</tr>
<tr>
<th class="grt-left">
x
</th>
<th class="grt-center grt-vrule-index" colspan="1">
Sub 1
</th>
<th class="grt-center grt-vrule-1" colspan="1">
Sub 2
</th>
<th class="grt-center grt-vrule-0" colspan="1">
Sub 2
</th>
<th class="grt-center grt-vrule-1" colspan="1">
Sub 1
</th>
<th class="grt-center grt-vrule-0" colspan="1">
Sub 1
</th>
</tr>
</thead>
<tbody>
<tr>
<td class="grt-left">
2020
</td>
<td class="grt-right grt-vrule-index">
100
</td>
<td class="grt-right grt-vrule-1">
1.000n
</td>
<td class="grt-right grt-vrule-0">
601.00
</td>
<td class="grt-center grt-vrule-1">
2025-03-12
</td>
<td class="grt-left grt-vrule-0">
once upon a time
</td>
</tr>
<tr>
<td class="grt-left grt-hrule-0">
2021
</td>
<td class="grt-right grt-hrule-0 grt-vrule-index">
105
</td>
<td class="grt-right grt-hrule-0 grt-vrule-1">
31.623u
</td>
<td class="grt-right grt-hrule-0 grt-vrule-0">
1,450.75
</td>
<td class="grt-center grt-hrule-0 grt-vrule-1">
2025-05-24
</td>
<td class="grt-left grt-hrule-0 grt-vrule-0">
lived happily ever after
</td>
</tr>
<tr>
<td class="grt-left grt-hrule-0">
2022
</td>
<td class="grt-right grt-hrule-0 grt-vrule-index">
2,000
</td>
<td class="grt-right grt-hrule-0 grt-vrule-1">
1.000
</td>
<td class="grt-right grt-hrule-0 grt-vrule-0">
2,300.50
</td>
<td class="grt-center grt-hrule-0 grt-vrule-1">
2025-08-06
</td>
<td class="grt-left grt-hrule-0 grt-vrule-0">
not in Kansas anymore
</td>
</tr>
<tr>
<td class="grt-left grt-hrule-0">
2023
</td>
<td class="grt-right grt-hrule-0 grt-vrule-index">
2,025
</td>
<td class="grt-right grt-hrule-0 grt-vrule-1">
31.623k
</td>
<td class="grt-right grt-hrule-0 grt-vrule-0">
3,150.25
</td>
<td class="grt-center grt-hrule-0 grt-vrule-1">
2025-10-18
</td>
<td class="grt-left grt-hrule-0 grt-vrule-0">
neutrinos are hard to detect
</td>
</tr>
<tr>
<td class="grt-left grt-hrule-0">
2024
</td>
<td class="grt-right grt-hrule-0 grt-vrule-index">
100,000
</td>
<td class="grt-right grt-hrule-0 grt-vrule-1">
1.000G
</td>
<td class="grt-right grt-hrule-0 grt-vrule-0">
4,000.00
</td>
<td class="grt-center grt-hrule-0 grt-vrule-1">
2025-12-31
</td>
<td class="grt-left grt-hrule-0 grt-vrule-0">
risk is hard to define
</td>
</tr>
</tbody>
</table>

 
Note the following features:

-   Text is left-aligned, numbers are right-aligned.
-   The index is displayed, was detected as likely years, and formatted
    without a comma separator.
-   The first column of integers does have a comma separator.
-   The second column of floats spans several orders of magnitude and is
    formatted using Engineering format, n for nano through G for giga.
-   The third column of floats is formatted with a comma separator and
    two decimals, based on the average absolute value.
-   The fourth column of date times is formatted as ISO standard dates
    (not date times).
-   The vertical lines separate the levels of the column multiindex. The
    subgroups are a little tricky.

More coming soon.

## Documentation

Available on
[readthedocs](https://greater-tables-project.readthedocs.io/en/latest).
