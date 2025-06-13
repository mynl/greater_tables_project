# coding: utf-8
"""
Define text table formats.

Based on model in tabulate.

Tabulate table formats are defined in namedtuples found in tf:

from tabulate import _table_formats as tf

This is a dict and e.g.,::

    tf['mixed_grid'] = TableFormat(
        lineabove=Line(begin='┍', hline='━', sep='┯', end='┑'),
        linebelowheader=Line(begin='┝', hline='━', sep='┿', end='┥'),
        linebetweenrows=Line(begin='├', hline='─', sep='┼', end='┤'),
        linebelow=Line(begin='┕', hline='━', sep='┷', end='┙'),
        headerrow=DataRow(begin='│', sep='│', end='│'),
        datarow=DataRow(begin='│', sep='│', end='│'),
        padding=1,
        with_header_hide=None)

    tf.keys() = dict_keys(['simple', 'plain', 'grid', 'simple_grid',
    'rounded_grid', 'heavy_grid', 'mixed_grid', 'double_grid',
    'fancy_grid', 'outline', 'simple_outline', 'rounded_outline',
    'heavy_outline', 'mixed_outline', 'double_outline', 'fancy_outline',
    'github', 'pipe', 'orgtbl', 'jira', 'presto', 'pretty', 'psql', 'rst',
    'mediawiki', 'moinmoin', 'youtrack', 'html', 'unsafehtml', 'latex',
    'latex_raw', 'latex_bookconfig.tabs', 'latex_longtable', 'tsv', 'textile',
    'asciidoc'])

Parameters:
    df: pandas.DataFrame
        The data to display. Should have index reset, but specify index_levels.
    data_col_widths: list[int]
        List of visible widths (excluding padding) for each column.
    data_col_aligns: list[str]
        Alignment specifiers per column: 'left', 'center', or 'right'.
    index_levels: int
        Number of columns at the start considered index columns (split visually).
    fmt: TableFormat
        Box-drawing configuration (defaults to myFormat).
"""

from collections import namedtuple


# specify text mode
Line = namedtuple('Line', ['begin', 'hline', 'sep', 'end', 'index_sep'])
DataRow = namedtuple('DataRow', ['begin', 'sep', 'end', 'index_sep'])
TableFormat = namedtuple('TableFormat', [
    'lineabove',
    'linebelowheader',
    'linebetweenrows',
    'linebelow',
    'headerrow',
    'datarow',
    'padding',
    'with_header_hide'
])

# generic text format
GT_Format = TableFormat(
    lineabove=Line('┍', '━', '┯', '┑', '┳'),
    linebelowheader=Line('┝', '━', '┿', '┥', '╋'),
    linebetweenrows=Line('├', '─', '┼', '┤', '╂'),
    linebelow=Line('┕', '━', '┷', '┙', '┻'),
    headerrow=DataRow('│', '│', '│', '┃'),
    datarow=DataRow('│', '│', '│', '┃'),
    padding=1,
    with_header_hide=None
)

# GT_Format = TableFormat(
#     lineabove=Line('\u250d', '\u2501', '\u252f', '\u2511', '\u2533'),
#     linebelowheader=Line('\u251d', '\u2501', '\u253f', '\u2525', '\u254b'),
#     linebetweenrows=Line('\u251c', '\u2500', '\u253c', '\u2524', '\u2502'),
#     linebelow=Line('\u2515', '\u2501', '\u2537', '\u2519', '\u253b'),
#     headerrow=DataRow('\u2502', '\u2502', '\u2502', '\u2503'),
#     datarow=DataRow('\u2502', '\u2502', '\u2502', '\u2503'),
#     padding=1,
#     with_header_hide=None
# )


def default_formatter(x):
    """



    """
