# table formatting again
from bs4 import BeautifulSoup
from io import StringIO
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype,\
    is_float_dtype   # , is_numeric_dtype
import sys
import warnings


# turn this fuck-test off
pd.set_option('future.no_silent_downcasting', True)


# GPT recommended approach
logger = logging.getLogger(__name__)
# Disable log propagation to prevent duplicates
logger.propagate = False
if logger.hasHandlers():
    # Clear existing handlers
    logger.handlers.clear()
# SET DEGBUUGER LEVEL
LEVEL = logging.ERROR    # DEBUG or INFO, WARNING, ERROR, CRITICAL
logger.setLevel(LEVEL)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(LEVEL)
formatter = logging.Formatter('%(asctime)s | %(levelname)s |  %(funcName)-15s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Logger Setup; module recompiled.')


class GT(object):
    """Create greater_tables."""

    def __init__(self, df, caption='',
                 aligners=None, integer_default_str='{x:,d}',
                 float_default_str='{x:,.3f}',
                 date_default_str='{x:%Y-%m-%d}',
                 ratio_cols=None, ratio_default_str='{x:.1%}',
                 cast_to_floats=True, hrule_widths=None, vrule_widths=None,
                 column_names=False ,
                 pef_precision=3, pef_lower=-3, pef_upper=6,
                 font_size=0.9, debug=False):
        """
        Create a greater_tables formatting object, setting defaults.

        Wraps AND COPIES the dataframe df. WILL NOT REFLECT CHANGES TO DF.

        Provides html, latex, and markdown output in quarto/Jupyter accessible manner.

        pef determines where engineering format is used; pef_precision

        other date format '{%H:%M:%S}'

        :param aligners: None or dict (type or colname) -> left | center | right
        :param formatters: None or dict type -> format function to override defaults.
        """
        self.df = df.copy(deep=True)   # the object being formatted
        if not column_names:
            self.df.columns.names = [None] * self.df.columns.nlevels
        self.df_id = f'T{id(df):x}'.upper()
        self.caption = caption +  (' (id: ' + self.df_id + ')' if debug else '')

        # determine ratio columns
        if ratio_cols is not None and np.any(self.df.columns.duplicated()):
            logger.warning('Ratio cols specified with non-unique column names: ignoring request.')
            self.ratio_cols = []
        else:
            if ratio_cols is None:
                self.ratio_cols = []
            elif ratio_cols == 'all':
                self.ratio_cols = [i for i in self.df.columns]
            elif ratio_cols == 'base' or ratio_cols == 'default':
                self.ratio_cols = ['max_LR', 'gross_LR', 'net_LR', 'ceded_LR', 'LR', 'COC', 'CoC', 'ROE']
            elif ratio_cols is not None and not isinstance(ratio_cols, (tuple, list)):
                self.ratio_cols = [ratio_cols]

        if cast_to_floats:
            for i, c in enumerate(self.df.columns):
                old_type = self.df.dtypes[c]
                if not np.any((is_integer_dtype(self.df.iloc[:, i]),
                               is_datetime64_any_dtype(self.df.iloc[:, i]))):
                    try:
                        self.df.iloc[:, i] = self.df.iloc[:, i].astype(float)
                        logger.debug(f'coerce {i}={c} from {old_type} to float')
                    except ValueError:
                        logger.debug(f'coercing {i}={c} from {old_type} to float FAILED')

        # now can determine types
        self.float_col_indices = []
        self.integer_col_indices = []
        self.date_col_indices = []
        self.object_col_indices = []
        # manage non-unique col names here
        logger.debug('FIGURING TYPES')
        for i in range(self.df.shape[1]):
            ser = self.df.iloc[:, i]
            if is_datetime64_any_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is DATE')
                self.date_col_indices.append(i)
            elif is_integer_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is INTEGER')
                self.integer_col_indices.append(i)
            elif is_float_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is FLOAT')
                self.float_col_indices.append(i)
            else:
                logger.debug(f'col {i} = {self.df.columns[i]} is OBJECT')
                self.object_col_indices.append(i)

        # figure out column and index alignment
        if aligners is not None and np.any(self.df.columns.duplicated()):
            logger.warning('aligners specified with non-unique column names: ignoring request.')
            aligners = None
        if aligners is None:
            # not using
            aligners = []
        self.df_aligners = []

        lrc = {'l': 'gt_left', 'r': 'gt_right', 'c': 'gt_center'}
        for i, c in enumerate(df.columns):
            if c in aligners:
                self.df_aligners.append(lrc.get(aligners[c], 'gt_center'))
            elif c in self.ratio_cols or i in self.float_col_indices or i in self.integer_col_indices:
                # number -> right
                self.df_aligners.append('gt_right')
            elif i in self.date_col_indices:
                # center dates, why not!
                self.df_aligners.append('gt_center')
            else:
                # all else, left
                self.df_aligners.append('gt_left')

        # do same for index - but klunky...
        # not actually sure you ever want right or centered?
        self.df_idx_aligners = []
        for ser in [self.df.index] if self.df.index.nlevels == 1 else (
                self.df.index.get_level_values(i) for i in
                range(self.df.index.nlevels)):
            if is_datetime64_any_dtype(ser):
                self.df_idx_aligners.append('gt_left')
            elif is_integer_dtype(ser):
                self.df_idx_aligners.append('gt_left')
            elif is_float_dtype(ser):
                self.df_idx_aligners.append('gt_left')
            else:
                self.df_idx_aligners.append('gt_left')

        # store defaults
        self.integer_default_str = integer_default_str
        self.float_default_str = float_default_str    # VERY rarely used; for floats in cols that are not floats
        self.date_default_str = date_default_str
        self.ratio_default_str = ratio_default_str
        self.pef_precision = pef_precision
        self.pef_lower = pef_lower
        self.pef_upper = pef_upper
        self._pef = None
        self.font_size = font_size
        self.hrule_widths = hrule_widths
        self.vrule_widths = vrule_widths
        # because of the problem of non-unique indexes use a list and
        # not a dict to pass the formatters to to_html
        self._df_formatters = None

        self.df_style = None
        self.df_html = None

    # define the default and easy formatters ===================================================
    def ratio(self, x):
        """Ratio formatter."""
        try:
            return self.ratio_default_str.format(x=x)
        except ValueError:
            return x

    def date(self, x):
        """Date formatter."""
        try:
            return self.date_default_str.format(x=x)
            # return f'{x:%Y-%m-%d}'  # f"{dt:%H:%M:%S}"
        except ValueError:
            return x

    def integer(self, x):
        """Integer formatter."""
        try:
            return self.integer_default_str.format(x=x)
        except ValueError:
            return x

    def default_formatter(self, x):
        """Universal formatter for other types."""
        try:
            i = int(x)
            f = float(x)
            if i == f:
                return self.integer_default_str.format(x=i)
            else:
                # TODo BEEF UP?
                return self.float_default_str.format(x=f)
        except (TypeError, ValueError):
            return str(x)

    def pef(self, x):
        """Pandas engineering format."""
        if self._pef is None:
            self._pef = pd.io.formats.format.EngFormatter(accuracy=self.pef_precision, use_eng_prefix=True)
        return self._pef(x)

    def make_float_formatter(self, ser):
        """
        Make a float formatter suitable for the Series ser.

        Obeys these rules:
        * All elements in the column are formatted consistently
        * ...

        TODO flesh out... at some point shd use pef?!

        """
        amean = ser.abs().mean()
        # mean = ser.mean()
        amn = ser.abs().min()
        amx = ser.abs().max()
        # smallest = ser.abs().min()
        # sd = ser.sd()
        # p10, p50, p90 = np.quantile(ser, [0.1, .5, 0.9], method='inverted_cdf')
        # pl = 10. ** self.pef_lower
        # pu = 10. ** self.pef_upper
        pl, pu = 10. ** self.pef_lower, 10. ** self.pef_upper
        if amean < 1:
            precision = 5
        elif amean < 10:
            precision = 3
        elif amean < 20000:
            precision = 2
        else:
            precision = 0
        fmt = f'{{x:,.{precision}f}}'
        logger.debug(f'{ser.name=}, {amean=}, {fmt=}')
        if amean < pl or amean > pu or amx / max(1, amn) > pu:
            # go with eng
            def ff(x):
                try:
                    return self.pef(x)
                except (ValueError, TypeError):
                    return str(x)
        else:
            def ff(x):
                try:
                    if x == int(x) and np.abs(x) < pu:
                        return f'{x:,.0f}.'
                    else:
                        return fmt.format(x=x)
                except (ValueError, TypeError):
                    return str(x)
        return ff

    @ property
    def df_formatters(self):
        """
        Make and return the list of formatters.

        Created one per column. Int, date, objects use defaults, but
        for float cols the formatter is created custom to the details of
        each column.
        """
        # because of non-unique indexes, index by position not name
        if self._df_formatters is None:
            self._df_formatters = []
            for i, c in enumerate(self.df.columns):
                # set a default, note here can have
                # non-unique index so work with position i
                if c in self.ratio_cols:
                    # print(f'{i} ratio')
                    self._df_formatters.append(self.ratio)
                elif i in self.date_col_indices:
                    self._df_formatters.append(self.date)
                elif i in self.integer_col_indices:
                    # print(f'{i} int')
                    self._df_formatters.append(self.integer)
                elif i in self.float_col_indices:
                    # trickier approach...
                    self._df_formatters.append(self.make_float_formatter(self.df.iloc[:, i]))
                else:
                    # print(f'{i} default')
                    self._df_formatters.append(self.default_formatter)
            # self._df_formatters is now a list of length equal to cols in df
            if len(self._df_formatters) != self.df.shape[1]:
                raise ValueError(f'Something wrong: {len(self._df_formatters)=} != {self.df.shape=}')
        return self._df_formatters

    def __repr__(self):
        """Basic representation."""
        return f"GreaterTable wrapping df {len(self.df)} rows, id {self.df_id}"

    def _repr_html_(self):
        """
        Apply format to self.df.

        ratio cols like in constructor
        """
        nindex = self.df.index.nlevels
        ncolumns = self.df.columns.nlevels
        ncols = self.df.shape[1]
        dt = self.df.dtypes

        # call pandas built-in html converter
        # no escape so tex works
        html = self.df.to_html(table_id=self.df_id, formatters=self.df_formatters,
                               index_names=True, escape=False)

        hrule_widths = self.hrule_widths
        if hrule_widths is None:
            if nindex > 1:
                hrule_widths = (1.5, 1.0, 0)
            else:
                hrule_widths = (0, 0, 0)
        vrule_widths = self.vrule_widths
        if vrule_widths is None:
            if nindex > 1:
                vrule_widths = (1.5, 1.0, 0)
            else:
                vrule_widths = (0, 0, 0)

        # start to build style
        style = []
        style.append('<style>')
        style.append(f'''
    #{self.df_id}  {{
    border-collapse: collapse;
    font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
    font-size: {self.font_size}em;
    width: auto;
    border: none;
    overflow: auto;    }}
    #{self.df_id} thead {{
        /* top and bottom of header */
        border-top: 2px solid #000;
        border-bottom: 2px solid #000;
        }}
    #{self.df_id} tbody {{
        /* bottom of body */ 
        border-bottom: 2px solid #000;
        }}
    #{self.df_id} thead tr:nth-of-type({ncolumns+1}) th:nth-of-type(-n + {nindex - 1}) {{
        /* separate index column names in bottom row of header, ncolumns = number of levels in columns */
        /* border-right: 0.5px solid #000; */
        }}
    #{self.df_id} thead th:nth-of-type({nindex+1}) {{
        /* separate column headers from index on left first */
        border-left: {vrule_widths[0]}px solid #000;
        }}
    #{self.df_id} thead tr th:nth-of-type(n+{nindex+2}) {{
        /* grid around column elements, may overwrite left vertical of columns */
        /* nindex + 2 because left and want to exclude the first column */
        border-left: {vrule_widths[1]}px solid #000;
        }}
    #{self.df_id} thead tr th:nth-of-type(n+{nindex+1}) {{
        /* grid below column elements */
        border-bottom: {hrule_widths[1]}px solid #000;
        }}
    #{self.df_id} tbody tr th:nth-of-type(-n + {nindex -1}) {{
        /* verticals in index columns; excluding the right most ?? if nindex=1?? */ 
        /* problem for sparse index entries */
        /* border-right: 0.5px solid #000; */
        }}
    #{self.df_id} tbody tr td:nth-of-type(n + 2)  {{
        /* verticals in the body; index cols are th not td so fixed from second */
        border-left: {vrule_widths[2]}px solid #000;
    }}
    #{self.df_id} tbody th  {{
        vertical-align: top;
    }}
    #{self.df_id} caption {{
        padding-top: 10px;
        padding-bottom: 4px;
        font-size: 1.1em;
        text-align: left;
        font-weight: bold;
        caption-side: top;
    }}
    #{self.df_id} .gt_body_column {{
        /* separate index from body (version 2!)  */
        border-left: {vrule_widths[0]}px solid #000;
    }}
    #{self.df_id} .gt_ruled_row_0 {{
        /* bold h lines to separate level 1 mindex groups */
        border-top: {hrule_widths[0]}px solid #000;
    }}
    #{self.df_id} .gt_ruled_row_1 {{
        /* bold h lines to separate level 2 mindex groups */
        border-top: {hrule_widths[1]}px solid #000;
    }}
    #{self.df_id} .gt_ruled_row_2 {{
        /* bold h lines to separate level 3 mindex groups */
        border-top: {hrule_widths[2]}px solid #000;
    }}
    #{self.df_id} .gt_left {{
        text-align: left;
    }}
    #{self.df_id} .gt_center {{
        text-align: center;
    }}
    #{self.df_id} .gt_right {{
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}
    #{self.df_id} .gt_head {{
        /* font-family: "Times New Roman", 'Courier New'; */
        font-size: {self.font_size}em;
    }}
    #{self.df_id} td, th {{
        /* top, right, bottom left cell padding */
        padding: 2px 10px 2px 10px;
    }}
''')
        # ================================================
        style.append('</style>\n')

        if len(style) > 2:
            style = '\n'.join(style)
        else:
            style = ''
        self.df_style = style
        self.df_rawhtml = html

        # now alter html using bs4
        soup = BeautifulSoup(html, 'html.parser')

        # center all heading rows
        for r in soup.select('thead tr'):
            for td in r.find_all('th'):
                td['class'] = 'gt_center gt_head'

        # figure which index level changes for h rules
        chg_level = self.changed_level()
        for i, r in zip(range(len(self.df)), soup.select('tbody tr')):
            # row rules: if not first row and new level 0
            if i and (i in chg_level):
                lv = chg_level.loc[i]
                r['class'] = f'gt_ruled_row_{lv}'
            # to handle sparse index you have to work from the right!
            for a, td in zip(self.df_idx_aligners[::-1], r.find_all('th')[::-1]):
                td['class'] = f'{a} gt_head'
                # td['style'] = 'text-align: left;'
            for a, td in zip(self.df_aligners, r.find_all('td')):
                td['class'] = a
            # find the body column
            td = r.find_all('td')[-ncols]
            if td is not None:
                # print(f'Adding body col to {td.text}')
                existing_classes = td.get('class', '')  # Get existing classes as a list, or an empty list if none
                td['class'] = existing_classes + ' gt_body_column'  # Append new classes

        if self.caption != "":
            table = soup.find('table')
            if table:
                c = soup.new_tag('caption')
                c.string = self.caption
                c['class'] = 'gt_caption'
                table.insert(0, c)

        # take out dataframe
        table = soup.find("table", {"class": "dataframe"})
        if table:
            del table["class"]
        html = soup.prettify()
        self.df_html = html  # after alteration
        # return
        logger.info('CREATED HTML STYLE')
        return style + html

    @property
    def html(self):
        return ('' if self.df_style  is None else self.df_style) + (
            '' if self.df_html is None else self.df_html)

    def _repr_latex_(self):
        """Generate a LaTeX tabular representation."""
        logger.info('CREATED LATEX STYLE')
        # latex = self.df.to_latex(caption=self.caption, formatters=self._df_formatters)
        latex = self.df_to_tikz()
        return latex

    def changed_level(self):
        """
        Return the level of index that changes with each row.

        Very ingenious GTP code with some SM enhancements.
        """
        # otherwise you alter the actual index
        idx = self.df.index.copy()
        idx.names = [i for i in range(idx.nlevels)]
        # Determine at which level the index changes
        index_df = idx.to_frame(index=False)  # Convert MultiIndex to a DataFrame
        # true / false match last row
        tf = index_df.ne(index_df.shift())
        # changes need at least one true
        tf = tf.loc[tf.any(axis=1)]
        level_changes = tf.idxmax(axis=1)
        return level_changes

    def df_to_tikz(self, float_format=None, tabs=None,
                   show_index=True, scale=0.717, column_sep=3 / 8, row_sep=1 / 8,
                   figure='figure', extra_defs='', hrule=None, equal=False,
                   vrule=None, post_process='', label='', caption='', latex=None,
                   sparsify=1, clean_index=False):
        """
        Write DataFrame to custom tikz matrix to allow greater control of
        formatting and insertion of horizontal divider lines

        Estimates tabs from text width of fields (not so great if includes TeX);
        manual override available. Tabs gives the widths of each field in
        em (width of M)

        Standard row height = 1.5em seems to work - set in meta

        first and last thick rules by default
        others below (Python, zero-based) row number, excluding title row

        keyword arguments : value (no newlines in value) escape back slashes!
        ``#keyword...`` rows ignored
        passed in as a string to facilitate using them with %%pmt?

        **Rules**

        * hrule at i means below row i of the table. (1-based) Top, bottom and below index lines
          are inserted automatically. Top and bottom lines are thicker.
        * vrule at i means to the left of table column i (1-based); there will never be a rule to the far
          right...it looks plebby; remember you must include the index columns!

        sparsify  number of cols of multi index to sparsify

        Issue: colunn with floats and spaces or missing causess problems (VaR, TVaR, EPD, mean and CV table)

        From great.pres_maker.df_to_tikz

        keyword args:

            scale           scale applied to whole table - default 0.717
            height          row height, rec. 1 (em)
            column_sep      col sep in em
            row_sep         row sep in em
            figure          table, figure or sidewaysfigure
            color           color for text boxes (helps debugging)
            extra_defs      TeX defintions and commands put at top of table, e.g., \\centering
            lines           lines below these rows, -1 for next to last row etc.; list of ints
            post_process    e.g., non-line commands put at bottom of table
            label
            latex           arguments after \begin{table}[latex]
            caption         text for caption

        Previous version see great.pres_maker
        Original version see: C:\\S\\TELOS\\CAS\\AR_Min_Bias\\cvs_to_md.py

        :param df:
        :param fn_out:
        :param float_format:
        :param tabs:
        :param show_index:
        :param scale:
        :param column_sep:
        :param row_sep:
        :param figure:
        :param color:
        :param extra_defs:
        :param lines:
        :param post_process:
        :param label:
        :param caption:
        :return:
        """
        # local variable
        df = self.df.copy()

# \\begin{{{figure}}}{latex}
        header = """
\\centering
\\footnotesize
{extra_defs}
\\begin{{tikzpicture}}[
    auto,
    transform shape,
    nosep/.style={{inner sep=0}},
    table/.style={{
        matrix of nodes,
        row sep={row_sep}em,
        column sep={column_sep}em,
        nodes in empty cells,
        nodes={{rectangle, scale={scale}, text badly ragged}},
"""
        # put draw=blue!10 or so in nodes to see the node

        footer = """
{post_process}

\\end{{tikzpicture}}
"""
# {caption}
# \\end{{{figure}}}

        # make a safe float format
        if float_format is None:
            wfloat_format = GT.default_float_format
        else:
            # If you pass in a lambda function it won't have error handling
            def _ff(x):
                try:
                    return float_format(x)
                except:
                    return x
            wfloat_format = _ff

        if clean_index:
            # don't use the index
            # but you do use the columns, this does both
            # logger.debug(df.columns)
            df = GT.clean_index(df)
            # logger.debug(df.columns)

        # index
        if show_index:
            if isinstance(df.index, pd.MultiIndex):
                nc_index = len(df.index.levels)
                # df = df.copy().reset_index(drop=False, col_level=df.columns.nlevels - 1)
            else:
                nc_index = 1
            # col_level puts the label at the bottom of the column m index.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                df = df.reset_index(drop=False, col_level=df.columns.nlevels - 1)
            if sparsify:
                if hrule is None:
                    hrule = set()
            for i in range(sparsify):
                df.iloc[:, i], rules = GT._sparsify(df.iloc[:, i])
                # print(rules, len(rules), len(df))
                # don't want lines everywhere
                if len(rules) < len(df) - 1:
                    hrule = set(hrule).union(rules)
        else:
            nc_index = 0

        if nc_index:
            if vrule is None:
                vrule = set()
            else:
                vrule = set(vrule)
            # to the left of... +1
            vrule.add(nc_index + 1)

        if isinstance(df.columns, pd.MultiIndex):
            nr_columns = len(df.columns.levels)
        else:
            nr_columns = 1
        logger.debug(f'rows in columns {nr_columns}, cols in index {nc_index}')

        # internal TeX code
        matrix_name = hex(abs(hash(str(df))))

        # note this happens AFTER you have reset the index...need to pass number of index columns
        colw, mxmn, tabs = GT.guess_column_widths(df, nc_index=nc_index, float_format=wfloat_format, tabs=tabs,
                                                  scale=scale, equal=equal)
        # print(colw, tabs)
        logger.debug(f'tabs: {tabs}')
        logger.debug(f'colw: {colw}')

        # alignment dictionaries
        ad = {'l': 'left', 'r': 'right', 'c': 'center'}
        ad2 = {'l': '<', 'r': '>', 'c': '^'}
        # guess alignments: TODO add dates?
        align = []
        for n, i in zip(df.columns, df.dtypes):
            x, n = mxmn[n]
            if x == n and len(align) == 0:
                align.append('l')
            elif i == object and x == n:
                align.append('c')
            elif i == object:
                align.append('l')
            else:
                align.append('r')
        logger.debug(align)

        # start writing
        sio = StringIO()
        if latex is None:
            latex = ''
        else:
            latex = f'[{latex}]'
        sio.write(header.format(figure=figure, extra_defs=extra_defs, scale=scale, column_sep=column_sep,
                                row_sep=row_sep, latex=latex))

        # table header
        # title rows, start with the empty spacer row
        i = 1
        sio.write(f'\trow {i}/.style={{nodes={{text=black, anchor=north, inner ysep=0, text height=0, text depth=0}}}},\n')
        for i in range(2, nr_columns + 2):
            sio.write(f'\trow {i}/.style={{nodes={{text=black, anchor=south, inner ysep=.2em, minimum height=1.3em, font=\\bfseries}}}},\n')

        # write column spec
        for i, w, al in zip(range(1, len(align) + 1), tabs, align):
            # average char is only 0.48 of M
            # https://en.wikipedia.org/wiki/Em_(gtypography)
            if i == 1:
                # first column sets row height for entire row
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, text height=0.9em, text depth=0.2em, '
                          f'inner xsep={column_sep}em, inner ysep=0, '
                          f'text width={max(2, 0.6 * w):.2f}em}},\n')
            else:
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, nosep, text width={max(2, 0.6 * w):.2f}em}},\n')
        # extra col to right which enforces row height
        sio.write(f'\tcolumn {i+1:>2d}/.style={{text height=0.9em, text depth=0.2em, nosep, text width=0em}}')
        sio.write('\t}]\n')

        sio.write("\\matrix ({matrix_name}) [table, ampersand replacement=\\&]{{\n".format(matrix_name=matrix_name))

        # body of table, starting with the column headers
        # spacer row
        nl = ''
        for cn, al in zip(df.columns, align):
            s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=' '))
        # include the blank extra last column
        sio.write('\\& \\\\\n')
        # write header rows  (again, issues with multi index)
        mi_vrules = {}
        sparse_columns = {}
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in range(len(df.columns.levels)):
                nl = ''
                sparse_columns[lvl], mi_vrules[lvl] = GT._sparsify_mi(df.columns.get_level_values(lvl),
                                                                      lvl == len(df.columns.levels) - 1)
                for cn, c, al in zip(df.columns, sparse_columns[lvl], align):
                    c = wfloat_format(c)
                    s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                    nl = '\\&'
                    sio.write(s.format(cell=c + '\\grtspacer'))
                # include the blank extra last column
                sio.write('\\& \\\\\n')
        else:
            nl = ''
            for c, al in zip(df.columns, align):
                c = wfloat_format(c)
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=c + '\\grtspacer'))
            sio.write('\\& \\\\\n')

        # write table entries
        for idx, row in df.iterrows():
            nl = ''
            for c, cell, al in zip(df.columns, row, align):
                cell = wfloat_format(cell)
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=cell))
                # if c=='p':
                #     print('COLp', cell, type(cell), s, s.format(cell=cell))
            sio.write('\\& \\\\\n')
        sio.write(f'}};\n\n')

        # decorations and post processing - horizontal and vertical lines
        nr, nc = df.shape
        # add for the index and the last row plus 1 for the added spacer row at the top
        nr += nr_columns + 1
        # always include top and bottom
        # you input a table row number and get a line below it; it is implemented as a line ABOVE the next row
        # function to convert row numbers to TeX table format (edge case on last row -1 is nr and is caught, -2
        # is below second to last row = above last row)
        # shift down extra 1 for the spacer row at the top
        def python_2_tex(x): return x + nr_columns + 2 if x >= 0 else nr + x + 3
        tb_rules = [nr_columns + 1, python_2_tex(-1)]
        if hrule:
            hrule = set(map(python_2_tex, hrule)).union(tb_rules)
        else:
            hrule = list(tb_rules)
        logger.debug(f'hlines: {hrule}')

        # why
        yshift = row_sep / 2
        xshift = -column_sep / 2
        descender_proportion = 0.25

        # top rule is special
        ls = 'thick'
        ln = 1
        sio.write(f'\\path[draw, {ls}] ({matrix_name}-{ln}-1.south west)  -- ({matrix_name}-{ln}-{nc+1}.south east);\n')

        for ln in hrule:
            ls = 'thick' if ln == nr + nr_columns + 1 else ('semithick' if ln == 1 + nr_columns else 'very thin')
            if ln < nr:
                # line above TeX row ln+1 that exists
                sio.write(f'\\path[draw, {ls}] ([yshift={-yshift}em]{matrix_name}-{ln}-1.south west)  -- '
                          f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')
            else:
                # line above row below bottom = line below last row
                # descenders are 200 to 300 below baseline
                ln = nr
                sio.write(f'\\path[draw, thick] ([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-1.base west)  -- '
                          f'([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-{nc+1}.base east);\n')

        # if multi index put in lines within the index TODO make this better!
        if nr_columns > 1:
            for ln in range(2, nr_columns + 1):
                sio.write(f'\\path[draw, very thin] ([xshift={xshift}em, yshift={-yshift}em]'
                          f'{matrix_name}-{ln}-{nc_index+1}.south west)  -- '
                          f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')

        written = set(range(1, nc_index + 1))
        if vrule:
            # to left of col, 1 based, includes index
            # write these first
            # TODO fix madness vrule is to the left, mi_vrules are to the right...
            ls = 'very thin'
            for cn in vrule:
                if cn not in written:
                    sio.write(f'\\path[draw, {ls}] ([xshift={xshift}em]{matrix_name}-1-{cn}.south west)  -- '
                              f'([yshift={-descender_proportion-yshift}em, xshift={xshift}em]{matrix_name}-{nr}-{cn}.base west);\n')
                    written.add(cn - 1)

        if len(mi_vrules) > 0:
            logger.debug(f'Generated vlines {mi_vrules}; already written {written}')
            # vertical rules for the multi index
            # these go to the RIGHT of the relevant column and reflect the index columns already
            # mi_vrules = {level of index: [list of vrule columns]
            # written keeps track of which vrules have been done already; start by cutting out the index columns
            ls = 'ultra thin'
            for k, cols in mi_vrules.items():
                # don't write the lowest level
                if k == len(mi_vrules) - 1:
                    break
                for cn in cols:
                    if cn in written:
                        pass
                    else:
                        written.add(cn)
                        top = k + 1
                        if top == 0:
                            sio.write(f'\\path[draw, {ls}] ([xshift={-xshift}em]{matrix_name}-{top}-{cn}.south east)  -- '
                                      f'([yshift={-descender_proportion-yshift}em, xshift={-xshift}em]{matrix_name}-{nr}-{cn}.base east);\n')
                        else:
                            sio.write(f'\\path[draw, {ls}] ([xshift={-xshift}em, yshift={-yshift}em]{matrix_name}-{top}-{cn}.south east)  -- '
                                      f'([yshift={-descender_proportion-yshift}em, xshift={-xshift}em]{matrix_name}-{nr}-{cn}.base east);\n')

        if label == '':
            lt = ''
            label = '}  % no label'
        else:
            lt = label
            label = f'\\label{{tab:{label}}}'
        if caption == '':
            if label != '':
                logger.warning(f'You have a label but no caption; the label {label} will be ignored.')
            caption = '% caption placeholder'
        else:
            caption = f'\\caption{{{caption} {label}}}'
        sio.write(footer.format(figure=figure, post_process=post_process, caption=caption))

        return sio.getvalue()

    @staticmethod
    def guess_column_widths(df, nc_index, float_format, tabs=None, scale=1, equal=False):
        """
        estimate sensible column widths for the dataframe [in what units?]

        :param df:
        :param nc_index: number of columns in the index...these are not counted as "data columns"
        :param float_format:
        :param tabs:
        :return:
            colw   affects how the table is printed in the md file (actual width of data elements)
            mxmn   affects aligmnent: are all columns the same width?
            tabs   affecets the actual output
        """

        # this
        # tabs from _tabs, an estimate column widths, determines the size of the table columns as displayed
        colw = dict.fromkeys(df.columns, 0)
        headw = dict.fromkeys(df.columns, 0)
        _tabs = []
        mxmn = {}
        nl = nc_index
        for i, c in enumerate(df.columns):
            # figure width of the column labels; if index c= str, if MI then c = tuple
            # cw is the width of the column header/title
            if type(c) == str:
                if i < nl:
                    cw = len(c)
                else:
                    # for data columns look at words rather than whole phrase
                    cw = max(map(len, c.split(' ')))
                    # logger.info(f'leng col = {len(c)}, longest word = {cw}')
            else:
                # could be float etc.
                try:
                    cw = max(map(lambda x: len(float_format(x)), c))
                except TypeError:
                    # not a MI, float or something
                    cw = len(str(c))
            headw[c] = cw
            # now figure the width of the elements in the column
            # mxmn is used to determine whether to center the column (if all the same size)
            if df.dtypes.iloc[i] == object:
                # wierdness here were some objects actually contain floats, str evaluates to NaN
                # and picks up width zero
                try:
                    # _ = list(map(lambda x: len(float_format(x)), df.iloc[:, i]))
                    _ = df.iloc[:, i].map(lambda x: len(float_format(x)))
                    colw[c] = _.max()
                    mxmn[c] = (_.max(), _.min())
                except:
                    e = sys.exc_info()[0]
                    print(c, 'ERROR', e)
                    logger.error(f'{c} error {e} DO SOMETHING ABOUT THIS...if it never occurs dont need the if')
                    colw[c] = df[c].str.len().max()
                    mxmn[c] = (df[c].str.len().max(), df[c].str.len().min())
            else:
                # _ = list(map(lambda x: len(float_format(x)), df[c]))
                _ = df.iloc[:, i].map(lambda x: len(float_format(x)))
                colw[c] = _.max()
                mxmn[c] = (_.max(), _.min())
            # debugging grief
            # if c == 'p':
            #     print(c, df[c], colw[c], mxmn[c], list(map(len, list(map(float_format, df[c])))))
        if tabs is None:
            # now know all column widths...decide what to do
            # are all the columns about the same width?
            data_cols = np.array([colw[k] for k in df.columns[nl:]])
            same_size = (data_cols.std() <= 0.1 * data_cols.mean())
            common_size = 0
            if same_size:
                common_size = int(data_cols.mean() + data_cols.std())
                logger.info(f'data cols appear same size = {common_size}')
            for i, c in enumerate(df.columns):
                if i < nl or not same_size:
                    # index columns
                    _tabs.append(int(max(colw[c], headw[c])))
                else:
                    # data all seems about the same width
                    _tabs.append(common_size)
            logger.info(f'Determined tab spacing: {_tabs}')
            if equal:
                # see if equal widths makes sense
                dt = _tabs[nl:]
                if max(dt) / sum(dt) < 4 / 3:
                    _tabs = _tabs[:nl] + [max(dt)] * (len(_tabs) - nl)
                    logger.info(f'Taking equal width hint: {_tabs}')
                else:
                    logger.info(f'Rejecting equal width hint')
            # look to rescale, shoot for width of 150 on 100 scale basis
            data_width = sum(_tabs[nl:])
            index_width = sum(_tabs[:nl])
            target_width = 150 * scale - index_width
            if data_width / target_width < 0.9:
                # don't rescale above 1:1 - don't want too large
                rescale = min(1 / scale, target_width / data_width)
                _tabs = [w if i < nl else w * rescale for i, w in enumerate(_tabs)]
                logger.info(f'Rescale {rescale} applied; tabs = {_tabs}')

            tabs = _tabs

        return colw, mxmn, tabs

    @staticmethod
    def _sparsify(col):
        """
        sparsify col values, col a pd.Series or dict, with items and accessor
        column results from a reset_index so has index 0,1,2... this is relied upon.
        """
        last = col[0]
        new_col = col.copy()
        rules = []
        for k, v in col[1:].items():
            if v == last:
                new_col[k] = ''
            else:
                last = v
                rules.append(k - 1)
                new_col[k] = v
        return new_col, rules

    @staticmethod
    def _sparsify_mi(mi, bottom_level=False):
        """
        as above for a multi index level, without the benefit of the index...
        really all should use this function
        :param mi:
        :param bottom_level: for the lowest level ... all values repeated, no sparsificaiton
        :return:
        """
        last = mi[0]
        new_col = list(mi)
        rules = []
        for k, v in enumerate(new_col[1:]):
            if v == last and not bottom_level:
                new_col[k + 1] = ''
            else:
                last = v
                rules.append(k + 1)
                new_col[k + 1] = v
        return new_col, rules

    @staticmethod
    def clean_name(n):
        """
        escape underscores for using a name in a DataFrame index

        :param n:
        :return:
        """
        try:
            if type(n) == str:
                # quote underscores that are not in dollars
                return '$'.join((i if n % 2 else i.replace('_', '\\_') for n, i in enumerate(n.split('$'))))
            else:
                return n
        except:
            return n

    # @staticmethod
    # def clean_underscores(s):
    #     """
    #     check s for unescaped _s
    #     returns true if all _ escaped else false
    #     :param s:
    #     :return:
    #     """
    #     return np.all([s[x.start() - 1] == '\\' for x in re.finditer('_', s)])

    @staticmethod
    def clean_index(df):
        """
        escape _ for columns and index
        whether multi or not

        !!! you can do this with a renamer...

        :param df:
        :return:
        """

        idx_names = df.index.names
        col_names = df.columns.names

        if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            df.columns = GT.clean_mindex_work(df.columns)
        else:
            df.columns = map(GT.clean_name, df.columns)

        # index
        if isinstance(df.index, pd.core.indexes.multi.MultiIndex):
            df.index = GT.clean_mindex_work(df.index)
        else:
            df.index = map(GT.clean_name, df.index)
        df.index.names = idx_names
        df.columns.names = col_names
        return df

    @staticmethod
    def clean_mindex_work(idx):
        for i, lv in enumerate(idx.levels):
            if lv.dtype == 'object':
                repl = map(GT.clean_name, lv)
                idx = idx.set_levels(repl, level=i)
        return idx

    @staticmethod
    def default_float_format(x, neng=3):
        """
        the endless quest for the perfect float formatter...

        tester::

            for x in 1.123123982398324723947 * 10.**np.arange(-23, 23):
                print(default_float_format(x))

        :param x:
        :return:
        """
        ef = pd.io.formats.format.EngFormatter(neng, True)
        try:
            if x == 0:
                ans = '0'
            elif 1e-3 <= abs(x) < 1e6:
                if abs(x) <= 10:
                    ans = f'{x:.3g}'
                elif abs(x) < 100:
                    ans = f'{x:,.2f}'
                elif abs(x) < 1000:
                    ans = f'{x:,.1f}'
                else:
                    ans = f'{x:,.0f}'
            else:
                ans = ef(x)
            return ans
        except ValueError as e:
            logger.debug(f'ValueError {e}')
            return str(x)
        except TypeError as e:
            logger.debug(f'TypeError {e}')
            return str(x)
        except AttributeError as e:
            logger.debug(f'AttributeError {e}')
            return str(x)

# class GT_ORIGINAL(object):

#     def __init__(self, aligners=None, formatters=None, ratio_cols=None,
#                  precision=3, pef_lower=-3, pef_upper=6, font_size=0.9):
#         """
#         Create a greater_tables formatting object, setting defaults.

#         :param aligners: None or dict type -> left | center | right
#         :param formatters: None or dict type -> format function
#         """
#         # set defaults
#         if aligners is None:
#             aligners = {}
#         _base_aligners = {'number': 'right', 'date': 'center', 'str': 'left',
#                           'object': 'left'}
#         self.aligner = _base_aligners.update(aligners)

#         if formatters is None:
#             formatters = {}
#         _base_formatters = {int: GT._integer, float: GT._float, 'else': GT._default}
#         self.formatter_cache = _base_formatters.update(formatters)

#         _base_ratios = ['max_LR', 'gross_LR', 'net_LR', 'ceded_LR', 'LR',
#                         'COC', 'CoC', 'ROE']
#         self.ratio_cols = _base_ratios if ratio_cols is None else ratio_cols

#         self.precision = precision
#         self.pef_lower = pef_lower
#         self.pef_upper = pef_upper
#         self._pef = None
#         self.font_size = font_size

#         # archive what is used
#         self.formatter_cache = LRUCache(maxsize=10)
#         self.aligner_cache = LRUCache(maxsize=10)
#         self.last_id = ''
#         self.last_style = None
#         self.last_html = None

#     @property
#     def html(self):
#         return self.last_style + self.last_html

#     # default formatters
#     @staticmethod
#     def _ratio(x):
#         try:
#             return f'{x:.1%}'
#         except:
#             return x

#     @staticmethod
#     def _date(x):
#         try:
#             return f'{x:%Y-%m-%d}'  # f"{dt:%H:%M:%S}"
#         except:
#             return x

#     @staticmethod
#     def _float(x):
#         try:
#             return f'{x:,.3f}'
#         except:
#             return x

#     @staticmethod
#     def _integer(x):
#         try:
#             return f'{x:,d}'
#         except:
#             return x

#     @staticmethod
#     def _default(x):
#         try:
#             i = int(x)
#             f = float(x)
#             if i == f:
#                 return f'{i:,d}'
#             else:
#                 return f'{i:,.3f}'
#         except ValueError:
#             return str(x)

#     def pef(self, x):
#         """Pandas engineering format."""
#         if self._pef is None:
#             self._pef = pd.io.formats.format.EngFormatter(accuracy=self.precision, use_eng_prefix=True)
#         return self._pef(x)

#     def make_float_formatter(self, ser):
#         """
#         Make a float formatter suitable for the Series ser.

#         Obeys these rules:
#         * All elements in the column are formatted consistently
#         * ...

#         """
#         amean = ser.abs().mean()
#         # mean = ser.mean()
#         # mn = ser.min()
#         # mx = ser.max()
#         # smallest = ser.abs().min()
#         # sd = ser.sd()
#         # p10, p50, p90 = np.quantile(ser, [0.1, .5, 0.9], method='inverted_cdf')
#         # pl = 10. ** self.pef_lower
#         # pu = 10. ** self.pef_upper
#         if amean < 1:
#             precision = 5
#         elif amean < 10:
#             precision = 3
#         elif amean < 20000:
#             precision = 2
#         else:
#             precision = 0
#         fmt = f'{{x:,.{precision}f}}'
#         logger.info(f'{ser.name=}, {amean=}, {fmt=}')

#         def ff(x):
#             try:
#                 if x == int(x):
#                     return f'{x:,.0f}.'
#                 else:
#                     return fmt.format(x=x)
#             except:
#                 return str(x)
#         return ff

#     @property
#     def last_formatter(self):  # noqa
#         return self.formatter_cache.get(self.last_id, 'None defined.')

#     @property
#     def last_aligner(self):  # noqa
#         return self.aligner_cache.get(self.last_id, 'None defined.')

#     def to_html(self, df, table_id, caption='', ratio_cols=None, **kwargs):
#         """
#         Convert to html with suitable number formatting.

#         Step 1 of apply format, separated so it can be called stand-alone.
#         """
#         if ratio_cols is not None and np.any(df.columns.duplicated()):
#             logger.warning('Ratio cols specified with non-unique column names: ignoring request.')
#         else:
#             if ratio_cols == 'all':
#                 ratio_cols = [i for i in df.columns]
#             elif ratio_cols == 'default':
#                 ratio_cols = self.ratio_cols
#             elif ratio_cols is not None and type(ratio_cols) != list:
#                 ratio_cols = [ratio_cols]
#             # check index valid
#             elif ratio_cols is None:
#                 ratio_cols = []

#         # because of non-unique indexes, index by position not
#         # name
#         float_cols = df.select_dtypes(include=["float64", "float32", "float"]).columns
#         float_cols = [j for j, i in enumerate(df.columns) if i in float_cols]
#         integer_cols = df.select_dtypes(include=["int64", "int32", "int"]).columns
#         integer_cols = [j for j, i in enumerate(df.columns) if i in integer_cols]
#         date_cols = df.select_dtypes(include=["datetime64"]).columns
#         date_cols = [j for j, i in enumerate(df.columns) if i in date_cols]

#         # because of the problem of non-unique indexes use a list and
#         # not a dict to pass the formatters to to_html
#         formatter_list = []

#         for i, c in enumerate(df.columns):
#             # set a default, note here can have
#             # non-unique index so work with position i
#             if c in ratio_cols:
#                 # print(f'{i} ratio')
#                 formatter_list.append(self._ratio)
#             elif i in date_cols:
#                 formatter_list.append(self._date)
#             elif i in integer_cols:
#                 # print(f'{i} int')
#                 formatter_list.append(self._integer)
#             elif i in float_cols:
#                 # trickier approach...
#                 formatter_list.append(self.make_float_formatter(df.iloc[:, i]))
#             else:
#                 # print(f'{i} default')
#                 formatter_list.append(self._default)
#         # logger.info(str(formatter_list))
#         # formatter_list is now a list of length equal to cols in df
#         assert len(formatter_list) == df.shape[1], f'Something wrong: {len(formatter_list)=} != {df.shape=}'
#         self.formatter_cache[table_id] = formatter_list
#         html = df.to_html(table_id=table_id, formatters=formatter_list, **kwargs)
#         return html

#     def apply_format(self, df, caption='', hrule_widths=None, ratio_cols=None, **kwargs):
#         """
#         Apply format to df.

#         ratio cols like in constructor
#         """
#         self.last_id = table_id = f'T{id(df):x}'[::2].upper()

#         nindex = df.index.nlevels
#         ncolumns = df.columns.nlevels
#         ncols = df.shape[1]

#         dt = df.dtypes

#         html = self.to_html(df, table_id, caption=caption, ratio_cols=ratio_cols, **kwargs)

#         # for now...
#         # guess: index l, numeric r rest l
#         idx = 'l' * df.index.nlevels
#         numeric_cols = df.select_dtypes('number').columns
#         rc = ''.join('r' if c in numeric_cols else 'l' for c in df.columns)
#         col_align = idx + rc

#         if hrule_widths is None:
#             if nindex > 1:
#                 hrule_widths = (1.5, 1.0, 0.5)
#             else:
#                 hrule_widths = (0, 0, 0)
#         # start to build style
#         style = []
#         style.append('<style>')
#         style.append(f'''#{table_id}  {{
#     border-collapse: collapse;
#     font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
#     font-size: {self.font_size}em;
#     /* line-height: normal;
#     margin-left: auto;
#     margin-right: auto; */
#     width: auto;
#     overflow: auto;    }}
#     #{table_id} thead {{
#         border-top: 2px solid #000;
#         border-bottom: 2px solid #000;
#         }}
#     #{table_id} tbody {{
#         border-bottom: 2px solid #000;
#         }}
#     #{table_id} thead tr:nth-of-type({ncolumns+1}) th:nth-of-type(-n + {nindex - 1}) {{
#         border-right: 0.5px solid #000;
#         }}
#     #{table_id} thead th:nth-of-type({nindex+1}) {{
#         border-left: 1.5px solid #000;
#         border-bottom: 0.5px solid #000;
#         }}
#     #{table_id} thead th:nth-of-type(n+{nindex+2}) {{
#         border-left: 0.5px solid #000;
#         border-bottom: 0.5px solid #000;
#         }}
#     #{table_id} thead th:nth-of-type({nindex}) {{
#         border-bottom: 0.5px solid #000;
#         }}
#     #{table_id} tbody tr th:nth-of-type(-n + {nindex - 1}) {{
#         border-right: 0.5px solid #000;
#         }}
#     /*
#     #{table_id} tbody tr th {{
#         border-bottom: 0.5px solid #00f;
#         }}
#     #{table_id} tbody tr td {{
#         border-bottom: 0.5px solid #f00;
#         }}
#     #{table_id} tbody td {{
#         padding-left: 10px;
#         padding-right: 10px;
#         }}
#     #{table_id} tbody tr td:nth-of-type(1)  {{
#         border-left: 1.5px solid #000;
#     }}
#     */
#     #T6F2FE tbody td:nth-of-type({nindex + 1})  {{
#         border-left: 1.5px solid #000;
#     }}
#     #{table_id} tbody tr td:nth-of-type(n + 2)  {{
#         border-left: 0.5px solid #000;
#     }}
#     #{table_id} tbody th  {{
#         vertical-align: top;
#     }}
#     #{table_id} caption {{
#         padding-top: 10px;
#         padding-bottom: 4px;
#         font-size: 1.1em;
#         text-align: left;
#         /* font-weight: bold; */
#         /* color: #f0f; */
#         }}
#     #{table_id} .gt_body_column {{
#         border-left: 1px solid #000;
#     }}
#     #{table_id} .gt_ruled_row_0 {{
#         border-top: {hrule_widths[0]}px solid #000;
#     }}
#     #{table_id} .gt_ruled_row_1 {{
#         border-top: {hrule_widths[0]}px solid #444;
#     }}
#     #{table_id} .gt_ruled_row_2 {{
#         border-top: {hrule_widths[0]}px solid #888;
#     }}
#     #{table_id} .gt_left {{
#         text-align: left;
#         }}
#     #{table_id} .gt_center {{
#         text-align: center;
#         }}
#     #{table_id} .gt_right {{
#         text-align: right;
#         font-variant-numeric: tabular-nums;
#         }}
#     #{table_id} .gt_head {{
#         /* font-family: "Times New Roman", 'Courier New'; */
#         font-size: {self.font_size}em;
#         }}
#     #{table_id} td, th {{
#       /* top, right, bottom left */
#       padding: 2px 10px 2px 10px;
#     }}
# ''')
#         # ================================================
#         style.append('</style>\n')

#         if len(style) > 2:
#             style = '\n'.join(style)
#         else:
#             style = ''
#         self.last_style = style
#         self.last_rawhtml = html

#         # now alter html using bs4
#         soup = BeautifulSoup(html, 'html.parser')

#         # center all heading rows
#         for r in soup.select('thead tr'):
#             for td in r.find_all('th'):
#                 td['class'] = 'gt_center gt_head'
#                 # td['style'] = 'text-align: left;'

#         # in each body row: headings left
#         alignment = []
#         for y in dt.values:
#             # if index is not unique this can return > 1 element
#             # know in same order as df.columns, so can use
#             # this approach
#             if y in (int, float):
#                 a = 'gt_right'
#             elif y == object:
#                 # print(c, 'object')
#                 a = 'gt_left'
#             else:
#                 # print(c, type(c), 'else')
#                 a = 'gt_center'
#             alignment.append(a)

#         idx_alignment = []
#         if nindex == 1:
#             idx_alignment.append(df.index.dtype)
#         else:
#             for c, t in df.index.dtypes.items():
#                 if t in (int, float):
#                     a = 'gt_right'
#                 elif t == object:
#                     a = 'gt_left'
#                 else:
#                     a = 'gt_center'
#                 idx_alignment.append(a)

#         # figure which index level changes for h rules
#         chg_level = GT.changed_level(df)
#         for i, r in zip(range(len(df)), soup.select('tbody tr')):
#             # row rules: if not first row and new level 0
#             if i and (i in chg_level):
#                 lv = chg_level.loc[i]
#                 r['class'] = f'gt_ruled_row_{lv}'
#             # to handle sparse index you have to work from the right!
#             for a, td in zip(idx_alignment[::-1], r.find_all('th')[::-1]):
#                 td['class'] = f'{a} gt_head'
#                 # td['style'] = 'text-align: left;'
#             for a, td in zip(alignment, r.find_all('td')):
#                 td['class'] = a
#             # find the body column
#             td = r.find_all('td')[-ncols]
#             if td is not None:
#                 # print(f'Adding body col to {td.text}')
#                 existing_classes = td.get('class', '')  # Get existing classes as a list, or an empty list if none
#                 td['class'] = existing_classes + ' gt_body_column'  # Append new classes
#         if caption != "":
#             table = soup.find('table')

#             # Add a caption
#             if table:
#                 c = soup.new_tag('caption')
#                 c.string = caption
#                 c['class'] = 'gt_caption'
#                 table.insert(0, c)

#         # take out dataframe
#         table = soup.find("table", {"class": "dataframe"})
#         if table:
#             del table["class"]

#         html = soup.prettify()
#         self.last_html = html  # after alteration
#         # return
#         return style + html

#     __call__ = apply_format

#     def apply_styles(self, html):
#         """
#         Apply css styles to the base table.

#         Whine.
#         """
#         pass

#     @staticmethod
#     def changed_level(df):
#         """
#         Return the level of index that changes with each row.

#         Very ingenious GTP code with some SM enhancements.
#         """
#         idx = df.index
#         idx.names = [i for i in range(idx.nlevels)]
#         # Determine at which level the index changes
#         index_df = idx.to_frame(index=False)  # Convert MultiIndex to a DataFrame
#         # true / false match last row
#         tf = index_df.ne(index_df.shift())
#         # changes need at least one true
#         tf = tf.loc[tf.any(axis=1)]
#         level_changes = tf.idxmax(axis=1)
#         return level_changes
