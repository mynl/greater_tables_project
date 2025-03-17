# table formatting again
from bs4 import BeautifulSoup
from decimal import InvalidOperation
from io import StringIO
from itertools import groupby
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype,\
    is_float_dtype   # , is_numeric_dtype
from pathlib import Path
import re
import sys
import warnings


from .hasher import df_short_hash

# turn this fuck-fest off
pd.set_option('future.no_silent_downcasting', True)
# pandas complaining about casting columns eg putting object in float column
warnings.simplefilter(action='ignore', category=FutureWarning)


# GPT recommended approach
logger = logging.getLogger(__name__)
# Disable log propagation to prevent duplicates
logger.propagate = False
if logger.hasHandlers():
    # Clear existing handlers
    logger.handlers.clear()
# SET DEGBUGGER LEVEL
LEVEL = logging.WARNING    # DEBUG or INFO, WARNING, ERROR, CRITICAL
logger.setLevel(LEVEL)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(LEVEL)
formatter = logging.Formatter('%(asctime)s | %(levelname)s |  %(funcName)-15s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(f'Logger Setup; {__name__} module recompiled.')


class GT(object):
    """Create greater_tables."""

    def __init__(self,
                 df,
                 caption='',
                 aligners=None,
                 ratio_cols=None,
                 year_cols=None,
                 show_index=True,
                 default_integer_str='{x:,d}',
                 default_float_str='{x:,.3f}',
                 default_date_str='%Y-%m-%d',
                 default_ratio_str='{x:.1%}',
                 table_float_format=None,
                 table_hrule_width=1,
                 table_vrule_width=1,
                 hrule_widths=None,
                 vrule_widths=None,
                 sparsify=True,             # index sparsification - almost certainly want this!
                 sparsify_columns=True,     # column sparsification with colspans
                 spacing='medium',          # tight, medium, wide
                 padding_trbl=None,         # tuple of four ints for padding
                 font_body=0.9,
                 font_head=1.0,
                 font_caption=1.1,
                 font_bold_index=False,
                 pef_precision=3,
                 pef_lower=-3,
                 pef_upper=6,
                 cast_to_floats=True,
                 header_row=True,
                 tabs=None,
                 equal=False,
                 debug=False):
        """
        Create a greater_tables formatting object.

        Provides html and latex output in quarto/Jupyter accessible manner.
        Wraps AND COPIES the dataframe df. WILL NOT REFLECT CHANGES TO DF.

        Recommended usage is to derive from GT and set defaults suitable to your particular application.
        In that way you can maintain a "house-style"

        :param df: target DataFrame or list of lists or markdown table string
        :param caption: table caption, optional
        :param aligners: None or dict (type or colname) -> left | center | right
        :param ratio_cols: None, or "all" or list of column names treated as ratios. Set defaults in derived class suitable to application.
        :param year_cols: None, or "all" or list of column names treated as years (no commas, no decimals). Set defaults in derived class suitable to application.
        :param show_index: if True, show the index columns, default True
        :param default_integer_str: format f-string for integers, default '{x:,d}'
        :param default_float_str: format f-string for floats, default '{x:,.3f}'
        :param default_date_str: format f-string for dates, default '%Y-%m-%d'. NOTE: no braces or x!
        :param default_ratio_str: format f-string for ratios, default '{x:.1%}'
        :param table_float_format: None or format string for floats in the table format function, applied to entire table, default None
        :param cast_to_floats: if True, try to cast all non-integer, non-date columns to floats
        :param table_hrule_width: width of the table top, botton and header hrule, default 1
        :param table_vrule_width: width of the table vrule, separating the index from the body, default 1
        :param hrule_widths: None or tuple of three ints for hrule widths (for use with multiindexes)
        :param vrule_widths: None or tuple of three ints for vrule widths (for use when columns have multiindexes)
        :param sparsify: if True, sparsify the index columns, you almost always want this to be true!
        :param sparsify_columns: if True, sparsify the columns, default True, generally a better look, headings centered in colspans
        :param spacing: 'tight', 'medium', 'wide' to quickly set cell padding. Medium is default (2, 10, 2, 10).
        :param padding_trbl: None or tuple of four ints for padding, in order top, right, bottom, left.
        :param font_body: font size for body text, default 0.9. Units in em.
        :param font_head: font size for header text, default 1.0. Units in em.
        :param font_caption: font size for caption text, default 1.1. Units in em.
        :param font_bold_index: if True, make the index columns bold, default False.
        :param pef_precision: precision (digits after period) for pandas engineering format, default 3.
        :param pef_lower: apply engineering format to floats with absolute value < 10**pef_lower; default -3.
        :param pef_upper: apply engineering format to floats with absolute value > 10**pef_upper; default 6.
        :param header_row: True: use first row as headers; False no headings. Default True
        :param tabs: None or list of column widths in characters or a common int or float width. (It is converted into em; one character is about 0.5em on average; digits are exactly 0.5em.) If None, will be calculated. Default None.
        :param equal: if True, set all column widths equal. Default False.
        :param debug: if True, add id to caption and use colored lines in table, default False.
        """
        # deal with alternative input modes
        if isinstance(df, pd.DataFrame):
            # usual use case
            pass
        elif isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, list):
            df = pd.DataFrame(df)
            # override this selection come what may
            show_index = False
            if header_row:
                # Set first row as column names
                df.columns = df.iloc[0]
                # Drop first row and reset index
                df = df[1:].reset_index(drop=True)
        elif isinstance(df, str):
            df, aligners = GT.md_to_df(df)
            show_index = False
        else:
            raise ValueError('df must be a DataFrame, a list of lists, or a markdown table string')

        if not df.columns.is_unique:
            raise ValueError('df column names are not unique')
        self.df = df.copy(deep=True)   # the object being formatted
        self.raw_df = df.copy(deep=True)
        # if not column_names:
        # get rid of column names
        # self.df.columns.names = [None] * self.df.columns.nlevels
        self.df_id = df_short_hash(self.df)
        self.debug = debug
        self.caption = caption + (' (id: ' + self.df_id + ')' if self.debug else '')

        # before messing
        self.show_index = show_index
        self.nindex = self.df.index.nlevels if self.show_index else 0
        self.ncolumns = self.df.columns.nlevels
        self.ncols = self.df.shape[1]
        self.dt = self.df.dtypes

        with warnings.catch_warnings():
            if self.show_index:
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                self.df = self.df.reset_index(drop=False, col_level=self.df.columns.nlevels - 1)
            # want the new index to be ints - that is not default if old was multiindex
            self.df.index = np.arange(self.df.shape[0], dtype=int)
        self.index_change_level = GT.changed_column(self.df.iloc[:, :self.nindex])
        if self.ncolumns > 1:
            # will be empty rows above the index headers
            self.index_change_level = pd.Series([i[-1] for i in self.index_change_level])

        self.column_change_level = GT.changed_level(self.raw_df.columns)

        # determine ratio columns
        if ratio_cols is not None and np.any(self.df.columns.duplicated()):
            logger.warning('Ratio cols specified with non-unique column names: ignoring request.')
            self.ratio_cols = []
        else:
            if ratio_cols is None:
                self.ratio_cols = []
            elif ratio_cols == 'all':
                self.ratio_cols = [i for i in self.df.columns]
            elif ratio_cols is not None and not isinstance(ratio_cols, (tuple, list)):
                self.ratio_cols = [ratio_cols]
            else:
                self.ratio_cols = ratio_cols

        # determine year columns
        if year_cols is not None and np.any(self.df.columns.duplicated()):
            logger.warning('Year cols specified with non-unique column names: ignoring request.')
            self.year_cols = []
        else:
            if year_cols is None:
                self.year_cols = []
            elif year_cols is not None and not isinstance(year_cols, (tuple, list)):
                self.year_cols = [year_cols]
            else:
                self.year_cols = year_cols

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            if cast_to_floats:
                for i, c in enumerate(self.df.columns):
                    old_type = self.df.dtypes[c]
                    if not np.any((is_integer_dtype(self.df.iloc[:, i]),
                                   is_datetime64_any_dtype(self.df.iloc[:, i]))):
                        try:
                            self.df.iloc[:, i] = self.df.iloc[:, i].astype(float)
                            logger.debug(f'coerce {i}={c} from {old_type} to float')
                        except (ValueError, TypeError):
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
        elif isinstance(aligners, str):
            # lrc for each column
            aligners = {c: a for c, a in zip(self.df.columns, aligners)}
        self.df_aligners = []

        lrc = {'l': 'grt-left', 'r': 'grt-right', 'c': 'grt-center'}
        # TODO: index aligners
        for i, c in enumerate(self.df.columns):
            # test aligners BEFORE index!
            if c in aligners:
                self.df_aligners.append(lrc.get(aligners[c], 'grt-center'))
            elif i < self.nindex:
                # index -> left
                self.df_aligners.append('grt-left')
            elif c in self.ratio_cols or i in self.float_col_indices or i in self.integer_col_indices:
                # number -> right
                self.df_aligners.append('grt-right')
            elif c in self.year_cols:
                self.df_aligners.append('grt-center')
            elif i in self.date_col_indices:
                # center dates, why not!
                self.df_aligners.append('grt-center')
            else:
                # all else, left
                self.df_aligners.append('grt-left')

        self.df_idx_aligners = self.df_aligners[:self.nindex]

        # store defaults
        self.default_integer_str = default_integer_str
        self.default_float_str = default_float_str    # VERY rarely used; for floats in cols that are not floats
        self.default_date_str = default_date_str.replace('{x:', '').replace('}', '')
        self.default_ratio_str = default_ratio_str
        self.pef_precision = pef_precision
        self.pef_lower = pef_lower
        self.pef_upper = pef_upper
        self._pef = None
        self.table_float_format = table_float_format
        self.default_float_formatter = None
        self.hrule_widths = hrule_widths or (0, 0, 0)
        self.vrule_widths = vrule_widths or (0, 0, 0)
        self.table_hrule_width = table_hrule_width
        self.table_vrule_width = table_vrule_width
        self.font_body = font_body
        self.font_head = font_head
        self.font_caption = font_caption
        self.font_bold_index = font_bold_index
        self.sparsify_columns = sparsify_columns
        if tabs is None:
            self.tabs = None
        elif isinstance(tabs, (int, float)):
            self.tabs = (tabs,)
        elif isinstance(tabs, (np.ndarray, list, tuple)):
            self.tabs = tabs  # Already iterable, self.tabs = as is
        else:
            self.tabs = [tabs]  # Fallback for anything else
        self.equal = equal

        if padding_trbl is None:
            if spacing == 'tight':
                padding_trbl = (0, 5, 0, 5)
            elif spacing == 'medium':
                padding_trbl = (2, 10, 2, 10)
            elif spacing == 'wide':
                padding_trbl = (4, 15, 4, 15)
            else:
                raise ValueError('spacing must be tight, medium, or wide or tuple of four ints.')
        try:
            self.padt, self.padr, self.padb, self.padl = padding_trbl
        except ValueError:
            logger.error(f'padding_trbl {padding_trbl=}, must be four ints, defaultign to medium')
            self.padt, self.padr, self.padb, self.padl = 2, 10, 2, 10

        # because of the problem of non-unique indexes use a list and
        # not a dict to pass the formatters to to_html
        self._df_formatters = None
        self.df_style = ''
        self.df_html = ''
        self._clean_html = ''
        self.tex = ''
        # finally sparsify and then apply formaters
        # this radically alters the df, so keep a copy for now...
        self.df_pre_applying_formatters = self.df.copy()
        self.df = self.apply_formatters(self.df)
        # sparsify
        if sparsify and self.nindex > 1:
            for c in self.df.columns[:self.nindex]:
                # spartify returns some other stuff...
                self.df[c], _ = GT.sparsify(self.df[c])

    # define the default and easy formatters ===================================================
    def default_ratio_formatter(self, x):
        """Ratio formatter."""
        try:
            return self.default_ratio_str.format(x=x)
        except ValueError:
            return str(x)

    def default_date_formatter(self, x):
        """Date formatter."""
        try:
            return x.strftime(self.default_date_str) if pd.notna(x) else ""
            # return self.default_date_str.format(x=x)
            # return f'{x:%Y-%m-%d}'  # f"{dt:%H:%M:%S}"
        except ValueError:
            logger.error(f'date error with {x=}')
            return str(x)

    def default_integer_formatter(self, x):
        """Integer formatter."""
        try:
            return self.default_integer_str.format(x=x)
        except ValueError:
            return str(x)

    def default_year_formatter(self, x):
        """Year formatter."""
        try:
            return f'{int(x):d}'
        except ValueError:
            return str(x)

    def default_formatter(self, x):
        """Universal formatter for other types."""
        try:
            f = float(x)
            if self.default_float_formatter:
                return self.default_float_formatter(f)
            try:
                i = int(x)
            except ValueError:
                i = int(f)
            if i == f:
                return self.default_integer_str.format(x=i)
            else:
                # TODO BEEF UP?
                return self.default_float_str.format(x=f)
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
                except (ValueError, TypeError, InvalidOperation):
                    return str(x)
        else:
            def ff(x):
                try:
                    return fmt.format(x=x)
                    # well and good but results in ugly differences
                    # by entries in a column
                    # if x == int(x) and np.abs(x) < pu:
                    #     return f'{x:,.0f}.'
                    # else:
                    #     return fmt.format(x=x)
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
        if self.table_float_format is not None:
            if callable(self.table_float_format):
                # wrap in error protections
                def ff(x):
                    try:
                        return self.table_float_format(x=x)
                    except ValueError:
                        return str(x)
                    except Exception as e:
                        logger.error(f'Custom float function raised {e=}')
                self.default_float_formatter = ff
            else:
                if type(self.table_float_format) != str:
                    raise ValueError('table_float_format must be a string or a function')
                fmt = self.table_float_format
                def ff(x):
                    try:
                        return fmt.format(x=x)
                    except ValueError:
                        return str(x)
                    except Exception as e:
                        logger.error(f'Custom float format string raised {e=}')
                self.default_float_formatter = ff
        else:
            self.default_float_formatter = False

        if self._df_formatters is None:
            self._df_formatters = []
            for i, c in enumerate(self.df.columns):
                # set a default, note here can have
                # non-unique index so work with position i
                if c in self.ratio_cols:
                    # print(f'{i} ratio')
                    self._df_formatters.append(self.default_ratio_formatter)
                elif c in self.year_cols:
                    self._df_formatters.append(self.default_year_formatter)
                elif i in self.date_col_indices:
                    self._df_formatters.append(self.default_date_formatter)
                elif i in self.integer_col_indices:
                    # print(f'{i} int')
                    self._df_formatters.append(self.default_integer_formatter)
                elif i in self.float_col_indices:
                    # trickier approach...
                    self._df_formatters.append(self.default_float_formatter or self.make_float_formatter(self.df.iloc[:, i]))
                else:
                    # print(f'{i} default')
                    self._df_formatters.append(self.default_formatter)
            # self._df_formatters is now a list of length equal to cols in df
            if len(self._df_formatters) != self.df.shape[1]:
                raise ValueError(f'Something wrong: {len(self._df_formatters)=} != {self.df.shape=}')
        return self._df_formatters

    def __repr__(self):
        """Basic representation."""
        return f"GreaterTable(df_id={self.df_id})"

    def _repr_html_(self):
        """
        Apply format to self.df.

        ratio cols like in constructor
        """
        return self.html

    def make_style(self, tabs):
        """Write out custom CSS for the table."""
        if self.debug:
            head_tb = '#0ff'
            body_b = '#f0f'
            h0 = '#f00'
            h1 = '#b00'
            h2 = '#900'
            bh0 = '#f00'
            bh1 = '#b00'
            v0 = '#0f0'
            v1 = '#0a0'
            v2 = '#090'
        else:
            head_tb = '#000'
            body_b = '#000'
            h0 = '#000'
            h1 = '#000'
            h2 = '#000'
            bh0 = '#000'
            bh1 = '#000'
            v0 = '#000'
            v1 = '#000'
            v2 = '#000'
        table_hrule = self.table_hrule_width
        table_vrule = self.table_vrule_width
        # for local use
        padt, padr, padb, padl = self.padt, self.padr, self.padb, self.padl

        style = [f'''
<style>
    #{self.df_id}  {{
    border-collapse: collapse;
    font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
    font-size: {self.font_body}em;
    width: auto;
    /* tb and lr 
    width: fit-content; */
    margin: 10px auto; 
    border: none;
    overflow: auto;
    margin-left: auto;
    margin-right: auto;
    }}
    /* tag formats */
    #{self.df_id} caption {{
        padding: {2 * padt}px {padr}px {padb}px {padl}px;
        font-size: {self.font_caption}em;
        text-align: center;
        font-weight: normal;
        caption-side: top;
    }}
    #{self.df_id} thead {{
        /* top and bottom of header */
        border-top: {table_hrule}px solid {head_tb};
        border-bottom: {table_hrule}px solid {head_tb};
        font-size: {self.font_head}em;
        }}
    #{self.df_id} tbody {{
        /* bottom of body */
        border-bottom: {table_hrule}px solid {body_b};
        }}
    #{self.df_id} th  {{
        vertical-align: bottom;
        padding: {2 * padt}px {padr}px {2 * padb}px {padl}px;
    }}
    #{self.df_id} td {{
        /* top, right, bottom left cell padding */
        padding: {padt}px {padr}px {padb}px {padl}px;
        vertical-align: top;
    }}
    /* class overrides */
    #{self.df_id} .grt-hrule-0 {{
        border-top: {self.hrule_widths[0]}px solid {h0};
    }}
    #{self.df_id} .grt-hrule-1 {{
        border-top: {self.hrule_widths[1]}px solid {h1};
    }}
    #{self.df_id} .grt-hrule-2 {{
        border-top: {self.hrule_widths[2]}px solid {h2};
    }}
    /* for the header, there if you have v lines you want h lines
       hence use vrule_widths */
    #{self.df_id} .grt-bhrule-0 {{
        border-bottom: {self.vrule_widths[0]}px solid {bh0};
    }}
    #{self.df_id} .grt-bhrule-1 {{
        border-bottom: {self.vrule_widths[1]}px solid {bh1};
    }}
    #{self.df_id} .grt-vrule-index {{
        border-left: {table_vrule}px solid {v0};
    }}
    #{self.df_id} .grt-vrule-0 {{
        border-left: {self.vrule_widths[0]}px solid {v0};
    }}
    #{self.df_id} .grt-vrule-1 {{
        border-left: {self.vrule_widths[1]}px solid {v1};
    }}
    #{self.df_id} .grt-vrule-2 {{
        border-left: {self.vrule_widths[2]}px solid {v2};
    }}
    #{self.df_id} .grt-left {{
        text-align: left;
    }}
    #{self.df_id} .grt-center {{
        text-align: center;
    }}
    #{self.df_id} .grt-right {{
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}
    #{self.df_id} .grt-head {{
        font-family: "Times New Roman", 'Courier New';
        font-size: {self.font_head}em;
    }}
    #{self.df_id} .grt-bold {{
        font-weight: bold;
    }}
''']
        for i, w in enumerate(tabs):
            style.append(f'    #{self.df_id} .grt-c-{i} {{ width: {w}em; }}')
        style.append('</style>')
        logger.info('CREATED CSS')
        return '\n'.join(style)

    def make_html(self):
        """Convert a pandas DataFrame to an HTML table."""
        index_name_to_level = dict(zip(self.raw_df.index.names, range(self.nindex)))
        index_change_level = self.index_change_level.map(index_name_to_level)
        # this is easier and computed in the init
        column_change_level = self.column_change_level

        # Start table
        html = [f'<table id="{self.df_id}">']
        if self.caption != '':
            html.append(f'<caption>{self.caption}</caption>')

        # Process header: allow_duplicates=True means can create cols with the same name
        bit = self.df.T.reset_index(drop=False, allow_duplicates=True)
        idx_header = bit.iloc[:self.nindex, :self.ncolumns]
        columns = bit.iloc[self.nindex:, :self.ncolumns]

        colw, tabs = GT.estimate_column_widths(self.df, nc_index=self.nindex, scale=1, equal=self.equal)
        if self.debug:
            print(f'Input {self.tabs=}\nComputed {tabs=}')
        if self.tabs is not None:
            if len(tabs) == len(self.tabs):
                tabs = self.tabs
            elif len(self.tabs) == 1:
                tabs = self.tabs * len(tabs)
            else:
                logger.error(f'{self.tabs=} must be None, a single number, or a list of numbers of the correct length. Ignoring.')
        # print('HTML ' + ', '.join([f'{c:,.2f}' for c in tabs]))

        # set column widths; tabs returns lengths of strings in each column
        # for proportional fonts, average char is 0.4 to 0.5 em but numbers with
        # tabular-nums are fixed 0.5, so use that
        # scale: want tables about 150-200 char wide, 1 char = 0.5 px size of font
        # so what 75-100 em wide in total
        # add the padding
        # TODO FONT SIZE
        # /4 works well for the tests (handles dates) but seems a bit illogical...
        tabs = np.array(tabs)  + (self.padl + self.padr) / 12  # guessing font size...
        # em_per_char = 0.5; true exactly for tabular-nums
        em_per_char = 0.6
        tabs = tabs * em_per_char
        # this gets stripped out by quarto, so make part of style
        html.append('<colgroup>')
        for w in tabs:
            html.append(f'<col style="width: {w * em_per_char}em;">')
        html.append('</colgroup>')

        # TODO Add header aligners
        # this is TRANSPOSED!!
        if self.sparsify_columns:
            html.append("<thead>")
            for i in range(self.ncolumns):
                # one per row of columns m index, usually only 1
                html.append("<tr>")
                if self.show_index:
                    for j, r in enumerate(idx_header.iloc[:, i]):
                        # columns one per level of index
                        html.append(f'<th class="grt-left">{r}</th>')
                # if not for col span issue you could just to this:
                # for j in range(self.ncols):
                #     hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
                #     if j == 0:
                #         # start with the first column come what may
                #         vrule = f'grt-vrule-index'
                #     elif j >= self.column_change_level[i]:
                #         vrule = f'grt-vrule-{column_change_level[cum_col]}'
                #     else:
                #         vrule = ''
                #     html.append(f'<th colspan="{colspan}" class="grt-center {hrule} {vrule}">{nm}</th>')
                # here, the groupby needs to consider all levels at and above i
                # this concats all the levels
                # need :i+1 to get down to the ith level
                cum_col = 0  # keep track of where we are up to
                for j, (nm, g) in enumerate(groupby(columns.iloc[:, :i+1].
                        apply(lambda x: ':::'.join(str(i) for i in x), axis=1))):
                    # ::: needs to be something that does not appear in the col names
                    # need to combine for groupby but be able to split off the last level
                    # picks off the name of the bottom level
                    nm = nm.split(':::')[-1]
                    hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
                    colspan = sum(1 for _ in g)
                    if 0 < j:
                        vrule = f'grt-vrule-{column_change_level[cum_col]}'
                    elif j == 0 and self.show_index:
                        # start with the first column if showing index
                        vrule = f'grt-vrule-index'
                    else:
                        vrule = ''
                    if j == 0 and not self.show_index:
                        # first column, no index, left align label
                        html.append(f'<th colspan="{colspan}" class="grt-left {hrule} {vrule}">{nm}</th>')
                    else:
                        html.append(f'<th colspan="{colspan}" class="grt-center {hrule} {vrule}">{nm}</th>')
                    cum_col += colspan
                html.append("</tr>")
            html.append("</thead>")
        else:
            html.append("<thead>")
            for i in range(self.ncolumns):
                # one per row of columns m index, usually only 1
                html.append("<tr>")
                if self.show_index:
                    for j, r in enumerate(idx_header.iloc[:, i]):
                        # columns one per level of index
                        html.append(f'<th class="grt-left">{r}</th>')
                for j, r in enumerate(columns.iloc[:, i]):
                    # one per column of dataframe
                    # figure how high up mindex the vrules go
                    # all headings get hrules, it's the vrules that are tricky
                    hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
                    if 0 < j < self.ncols and i >= column_change_level[j]:
                        vrule = f'grt-vrule-{column_change_level[j]}'
                    elif j == 0 and self.show_index:
                        # start with the first column come what may
                        vrule = f'grt-vrule-index'
                    else:
                        vrule = ''
                    html.append(f'<th class="grt-center {hrule} {vrule}">{r}</th>')
                html.append("</tr>")
            html.append("</thead>")

        bold_idx = 'grt-bold' if self.font_bold_index else ''
        html.append("<tbody>")
        for i, (n, r) in enumerate(self.df.iterrows()):
            # one per row of dataframe
            html.append("<tr>")
            hrule = ''
            if self.show_index:
                for j, c in enumerate(r.iloc[:self.nindex]):
                    # dx = data in index
                    # if this is the level that changes for this row
                    # will use a top rule  hence omit i = 0 which already has an hrule
                    if i > 0 and hrule == '' and j == index_change_level[i]:
                        hrule = f'grt-hrule-{j}'
                    # html.append(f'<td class="grt-dx-r-{i} grt-dx-c-{j} {self.df_aligners[j]} {hrule}">{c}</td>')
                    col_id = f'grt-c-{j}'
                    html.append(f'<td class="{col_id} {bold_idx} {self.df_aligners[j]} {hrule}">{c}</td>')
            for j, c in enumerate(r.iloc[self.nindex:]):
                # first col left handled by index/body divider
                if 0 < j < self.ncols:
                    vrule = f'grt-vrule-{column_change_level[j]}'
                elif j == 0 and self.show_index:
                    # start with the first column come what may
                    vrule = f'grt-vrule-index'
                else:
                    vrule = ''
                # html.append(f'<td class="grt-data-r-{i} grt-data-c-{j} {self.df_aligners[j+self.nindex]} {hrule} {vrule}">{c}</td>')
                col_id = f'grt-c-{j+self.nindex}'
                html.append(f'<td class="{col_id} {self.df_aligners[j+self.nindex]} {hrule} {vrule}">{c}</td>')
            html.append("</tr>")
        html.append("</tbody>")
        text = '\n'.join(html)
        self.df_html = GT.clean_html_tex(text)
        logger.info('CREATED HTML')
        self.df_style = self.make_style(tabs)

        return self.df_html

    def clean_style(self, soup):
        """Minify CSS inside <style> blocks and remove /* ... */ comments."""
        if not self.debug:
            for style_tag in soup.find_all("style"):
                if style_tag.string:
                    # Remove CSS comments
                    cleaned_css = re.sub(r'/\*.*?\*/', '', style_tag.string, flags=re.DOTALL)
                    # Minify whitespace
                    # cleaned_css = re.sub(r'\s+', ' ', cleaned_css).strip()
                    style_tag.string.replace_with(cleaned_css)
        return soup

    @property
    def html(self):
        if self._clean_html == '':
            if self.df_html == '':
                # makes style and html (need tabs)
                self.df_html = self.make_html()
            code = ["<div class='greater-table'>",
                    self.df_style,
                    self.df_html,
                    "</div>"]
            soup = BeautifulSoup('\n'.join(code), 'html.parser')
            soup = self.clean_style(soup)
            self._clean_html = str(soup) # .prettify() -> too many newlines
        return self._clean_html

    def _repr_latex_(self):
        """Generate a LaTeX tabular representation."""
        # return ''
        # latex = self.df.to_latex(caption=self.caption, formatters=self._df_formatters)
        latex = self.make_tikz()
        logger.info('CREATED LATEX STYLE')
        return latex

    @staticmethod
    def changed_column(bit):
        """Return the column that changes with each row."""
        tf = bit.ne(bit.shift())
        tf = tf.loc[tf.any(axis=1)]
        return tf.idxmax(axis=1)

    @staticmethod
    def changed_level(idx):
        """
        Return the level of index that changes with each row.

        Very ingenious GTP code with some SM enhancements.
        """
        # otherwise you alter the actual index
        idx = idx.copy()
        idx.names = [i for i in range(idx.nlevels)]
        # Determine at which level the index changes
        index_df = idx.to_frame(index=False)  # Convert MultiIndex to a DataFrame
        # true / false match last row
        tf = index_df.ne(index_df.shift())
        # changes need at least one true
        tf = tf.loc[tf.any(axis=1)]
        level_changes = tf.idxmax(axis=1)
        return level_changes

    @staticmethod
    def apply_formatters_work(df, formatters):
        """Apply formatters to a DataFrame."""
        new_df = pd.DataFrame({i: map(f, df.iloc[:, i])
                               for i, f in enumerate(formatters)})
        new_df.columns = df.columns
        return new_df

    def apply_formatters(self, df, mode='adjusted'):
        """
        Replace df (the raw df) with formatted df, including the index.

        If mode is 'adjusted' operates on columns only, does not touch the
        index. Otherwise, called from tikz and operating on raw_df
        """
        if mode == 'adjusted':
            # apply to df where the index has been reset
            # number of columns = len(self.df_formatters)
            return GT.apply_formatters_work(df, self.df_formatters)
        elif mode == 'raw':
            # work on raw_df where the index has not been reset
            # because of non-unique indexes, index by position not name
            # create the df and the index separately
            data_formatters = self.df_formatters[self.nindex:]
            new_body = GT.apply_formatters_work(df, data_formatters)
            if not self.show_index:
                return new_body
            # else have to handle the index
            index_formatters = self.df_formatters[:self.nindex]
            df_index = df.reset_index(drop=False, col_level=self.df.columns.nlevels - 1).iloc[:, :self.nindex]
            new_index = GT.apply_formatters_work(df_index, index_formatters)
            # put them back together
            new_df = pd.concat([new_index, new_body], axis=1)
            new_df = new_df.set_index(list(df_index.columns))
            new_df.index.names = df.index.names
            return new_df
        else:
            raise ValueError(f'unknown mode {mode}')

    def make_tikz(self,
                  scale=0.635,
                  column_sep=3 / 8,
                  row_sep=1 / 8,
                  figure='figure',
                  extra_defs='',
                  hrule=None,
                  vrule=None,
                  post_process='',
                  label='',
                  latex=None,
                  sparsify=1):
        """
        Write DataFrame to custom tikz matrix to allow greater control of
        formatting and insertion of horizontal and vertical divider lines

        Estimates tabs from text width of fields (not so great if includes
        a lot of TeX macros) with a manual override available. Tabs gives
        the widths of each field in em (width of M)

        Standard row height = 1.5em seems to work - set in meta.

        first and last thick rules
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

        Issue: column with floats and spaces or missing causes problems (VaR, TVaR, EPD, mean and CV table)

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
        # local variable - with all formatters already applied
        df = self.apply_formatters(self.raw_df.copy(), mode='raw')
        caption = ''
        if not df.columns.is_unique:
            # possible index/body column interaction
            raise ValueError('tikz routine requires unique column names')
# \\begin{{{figure}}}{latex}
# \\centering
# {extra_defs}
        # centering handled by quarto
        header = """
\\begin{{tikzpicture}}[
    auto,
    transform shape,
    nosep/.style={{inner sep=0}},
    table/.style={{
        matrix of nodes,
        row sep={row_sep}em,
        column sep={column_sep}em,
        nodes in empty cells,
        nodes={{rectangle, scale={scale}, text badly ragged {debug}}},
"""
        # put draw=blue!10 or so in nodes to see the node

        footer = """
{post_process}

\\end{{tikzpicture}}
"""
# {caption}
# \\end{{{figure}}}

        # always a good idea to do this...need to deal with underscores, %
        # and it handles index types that are not strings
        df = GT.clean_index(df)
        if not np.all([i=='object' for i in df.dtypes]):
            print('cols of df not all objects: ', df.dtypes, sep='\n')
        # make sure percents are escaped, but not if already escaped
        df = df.replace(r"(?<!\\)%", r"\%", regex=True)

        # pres_maker.df_to_tikz code line 1931
        if self.show_index:
            nc_index = df.index.nlevels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                df = df.reset_index(drop=False, col_level=df.columns.nlevels - 1)
            if sparsify:
                if hrule is None:
                    hrule = set()
            for i in range(sparsify):
                df.iloc[:, i], rules = GT.sparsify(df.iloc[:, i])
                # don't want lines everywhere
                if len(rules) < len(df) - 1:
                    hrule = set(hrule).union(rules)
        else:
            nc_index = 0

        if vrule is None:
            vrule = set()
        else:
            vrule = set(vrule)
        # to the left of... +1
        vrule.add(nc_index + 1)

        nr_columns = df.columns.nlevels
        logger.info(f'rows in columns {nr_columns}, columns in index {nc_index}')

        # internal TeX code (same as HTML code)
        matrix_name = self.df_id

        # note this happens AFTER you have reset the index...need to pass
        # number of index columns
        # have also converted everything to formatted strings
        # estimate... originally called guess_column_widths, with more parameters
        colw, tabs = GT.estimate_column_widths(df, nc_index=nc_index, scale=scale, equal=self.equal)
        if self.debug:
            print(f'Input {self.tabs=}\nComputed {tabs=}')
        if self.tabs is not None:
            if len(tabs) == len(self.tabs):
                tabs = self.tabs
            elif len(self.tabs) == 1:
                tabs = self.tabs * len(tabs)
            else:
                logger.error(f'{self.tabs=} must be None, a single number, or a list of numbers of the correct length. Ignoring.')
        # print('TIKZ ' + ', '.join([f'{c:,.2f}' for c in tabs]))
        # print(f'TIKZ {colw=}, {tabs=}')
        logger.info(f'tabs: {tabs}')
        logger.info(f'colw: {colw}')

        # alignment dictionaries - these are still used below
        ad = {'l': 'left', 'r': 'right', 'c': 'center'}
        ad2 = {'l': '<', 'r': '>', 'c': '^'}
        #  use df_aligners, at this point the index has been reset
        align = []
        for n, i in zip(df.columns, self.df_aligners):
            if i == 'grt-left':
                align.append('l')
            elif i == 'grt-right':
                align.append('r')
            elif i == 'grt-center':
                align.append('c')
            else:
                align.append('l')

        # start writing
        sio = StringIO()
        if latex is None:
            latex = ''
        else:
            latex = f'[{latex}]'
        debug = ''
        if self.debug:
            # color all boxes
            debug = ', draw=blue!10'
        sio.write(header.format(figure=figure, extra_defs=extra_defs, scale=scale, column_sep=column_sep,
                                row_sep=row_sep, latex=latex, debug=debug))

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
                sparse_columns[lvl], mi_vrules[lvl] = GT.sparsify_mi(df.columns.get_level_values(lvl),
                                                                     lvl == len(df.columns.levels) - 1)
                for cn, c, al in zip(df.columns, sparse_columns[lvl], align):
                    # c = wfloat_format(c)
                    s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                    nl = '\\&'
                    sio.write(s.format(cell=c + '\\grtspacer'))
                # include the blank extra last column
                sio.write('\\& \\\\\n')
        else:
            nl = ''
            for c, al in zip(df.columns, align):
                # c = wfloat_format(c)
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=c + '\\grtspacer'))
            sio.write('\\& \\\\\n')

        # write table entries
        for idx, row in df.iterrows():
            nl = ''
            for c, cell, al in zip(df.columns, row, align):
                # cell = wfloat_format(cell)
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
        if vrule and self.show_index:
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
            if lt != '':
                logger.info(f'You have a label but no caption; the label {label} will be ignored.')
            caption = '% caption placeholder'
        else:
            caption = f'\\caption{{{self.caption} {label}}}'
        sio.write(footer.format(figure=figure, post_process=post_process, caption=caption))

        self.tex = sio.getvalue()
        return self.tex

    @staticmethod
    def estimate_column_widths(df, nc_index, scale=1, equal=False):
        """
        Estimate sensible column widths for the dataframe [in what units?]

        Internal variables:
            mxmn   affects alignment: are all columns the same width?

        :param df:
        :param nc_index: number of columns in the index...these are not counted as "data columns"
        :param float_format:
        :param tabs:
        :param equal:  if True, try to make all data columns the same width (hint can be rejected)
        :return:
            colw   affects how the tex is printed to ensure it "looks neat" (actual width of data elements)
            tabs   affects the actual output
        """
        # this
        # tabs from _tabs, an estimate column widths, determines the size of the table columns as displayed
        # print(f'{nc_index=}, {scale=}, {equal=}')
        colw = dict.fromkeys(df.columns, 0)
        headw = dict.fromkeys(df.columns, 0)
        tabs = []
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
                # column name could be float etc. or if multi index a tuple
                try:
                    if isinstance(c, tuple):
                        # multiindex: join and split into words and take length of each word
                        words = ' '.join(c).split(' ')
                        cw = max(map(lambda x: len(str(x)), words))
                    else:
                        cw = max(map(lambda x: len(str(x)), c))
                    # print(f'{c}: {cw=} no error')
                except TypeError:
                    # not a MI, float or something
                    cw = len(str(c))
                    # print(f'{c}: {cw=} WITH error')
            headw[c] = cw
            # now figure the width of the elements in the column
            # mxmn is used to determine whether to center the column (if all the same size)
            if df.dtypes.iloc[i] == object:
                # wierdness here were some objects actually contain floats, str evaluates to NaN
                # and picks up width zero
                try:
                    lens = df.iloc[:, i].map(lambda x: len(str(x)))
                    colw[c] = lens.max()
                    mxmn[c] = (lens.max(), lens.min())
                except Exception as e:
                    logger.error(f'{c} error {e} DO SOMETHING ABOUT THIS...if it never occurs dont need the if')
                    colw[c] = df[c].str.len().max()
                    mxmn[c] = (df[c].str.len().max(), df[c].str.len().min())
            else:
                lens = df.iloc[:, i].map(lambda x: len(str(x)))
                colw[c] = lens.max()
                mxmn[c] = (lens.max(), lens.min())
            # print(f'{headw[c]=}, {colw[c]=}, {mxmn[c]=}, {c=}')
        # now know all column widths...decide what to do
        # are all the data columns about the same width?
        data_cols = np.array([colw[k] for k in df.columns[nl:]])
        same_size = (data_cols.std() <= 0.1 * data_cols.mean())
        # print(f'same size test requires {data_cols.std()} <= {0.1 * data_cols.mean()}')
        common_size = 0
        if same_size:
            common_size = int(data_cols.mean() + data_cols.std())
            logger.info(f'data cols appear same size = {common_size}')
            # print(f'data cols appear same size = {common_size}')
        for i, c in enumerate(df.columns):
            if i < nl or not same_size:
                # index columns
                tabs.append(int(max(colw[c], headw[c])))
            else:
                # data all seems about the same width
                tabs.append(common_size)
        logger.info(f'Determined tab spacing: {tabs}')
        if equal:
            # see if equal widths makes sense
            dt = tabs[nl:]
            if max(dt) / sum(dt) < 4 / 3:
                tabs = tabs[:nl] + [max(dt)] * (len(tabs) - nl)
                logger.info(f'Taking equal width hint: {tabs}')
                # print(f'Taking equal width hint: {tabs}')
            else:
                logger.info(f'Rejecting equal width hint')
                # print(f'Rejecting equal width hint')
        # look to rescale, shoot for width of 150 on 100 scale basis
        data_width = sum(tabs[nl:])
        index_width = sum(tabs[:nl])
        target_width = 150 * scale - index_width
        if data_width / target_width < 0.9:
            # don't rescale above 1:1 - don't want too large
            rescale = min(1 / scale, target_width / data_width)
            tabs = [w if i < nl else w * rescale for i, w in enumerate(tabs)]
            logger.info(f'Rescale {rescale} applied; tabs = {tabs}')
            # print(f'Rescale {rescale} applied; tabs = {tabs}')
        # print(f'{colw.values()=}\n{tabs=}')
        return colw, tabs

    @staticmethod
    def sparsify(col):
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
    def sparsify_mi(mi, bottom_level=False):
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
        Escape underscores for using a name in a DataFrame index
        and converts to a string. Also escape %.

        Called by Tikz routines.

        :param n:
        :return:
        """
        try:
            if type(n) == str:
                # quote underscores that are not in dollars
                return '$'.join((i if n % 2 else i.replace('_', '\\_').replace('%', '\\%')
                                 for n, i in enumerate(n.split('$'))))
            else:
                # can't contain an underscore!
                return str(n)
        except:
            return str(n)

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
    def clean_html_tex(text):
        """
        Clean TeX entries in HTML: $ -> \( and \) and $$ to \[ \].

        Apply after all other HTML rendering steps. HTML rendering only.
        """
        text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', text, flags=re.DOTALL)
        # Convert inline math: $...$ → \(...\)
        text = re.sub(r'(?<!\$)\$(.*?)(?<!\\)\$(?!\$)', r'\\(\1\\)', text)
        return text

    @staticmethod
    def clean_index(df):
        """
        escape _ for columns and index, being careful about subscripts
        in TeX formulas.

        :param df:
        :return:
        """
        return df.rename(index=GT.clean_name, columns=GT.clean_name)

    @staticmethod
    def default_float_format(x, neng=3):
        """
        the endless quest for the perfect float formatter...
        NOT USED AT THE MINUTE.

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

    def save_html(self, fn):
        """Save HTML to file."""
        p = Path(fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p = p.with_suffix('.html')
        soup = BeautifulSoup(self.html, 'html.parser')
        p.write_text(soup.prettify(), encodign='utf-8')
        logger.info(f'Saved to {p}')

    @staticmethod
    def md_to_df(txt):
        """Convert markdown text string table to DataFrame."""
        # remove starting and ending | in each line (optional anyway)
        txt = re.sub(r'^\||\|$', '', txt, flags=re.MULTILINE)
        txt = txt.strip().replace('*', '').split('\n')
        # remove the alignment row
        alignment_row = txt.pop(1)
        aligners = []
        for t in alignment_row.split('|'):
            if t[0] == ':' and t[-1] == ':':
                aligners.append('c')
            elif t[0] == ':':
                aligners.append('l')
            elif t[-1] == ':':
                aligners.append('r')
            else:
                # no alignment info
                pass
        if len(aligners) == 0:
            aligners = None
        else:
            aligners = ''.join(aligners)
        txt = [[j.strip()  for j in i.split('|')]  for i in txt]
        df = pd.DataFrame(txt).T
        df = df.set_index(0)
        df = df.T
        return df, aligners


class sGT(GT):
    """
    Example standard GT with Steve House-Style defaults.

    Each application can create its own defaults by subclassing GT
    in this way.
    """
    def __init__(self, df, caption="", guess_years=True, ratio_regex='lr|roe|coc', **kwargs):
        """Create Steve House-Style Formatter. Does not handle list of lists input."""
        if isinstance(df, str):
            df, aligners_ = GT.md_to_df(df)
            if 'aligners' not in kwargs:
                kwargs['aligners'] = aligners_
                kwargs['show_index'] = False

        nindex = df.index.nlevels
        ncolumns = df.columns.nlevels
        if 'ratio_cols' in kwargs:
            ratio_cols = kwargs['ratio_cols']
        else:
            if ratio_regex != '' and ncolumns == 1:
                ratio_cols = df.filter(regex=ratio_regex).columns.to_list()
            else:
                ratio_cols = None

        if guess_years:
            year_cols = sGT.guess_years(df)
        else:
            year_cols = kwargs.get('year_cols', None)

        # rule sizes
        hrule_widths = (1.5, 1, 0) if nindex > 1 else None
        vrule_widths = (1.5, 1, 0.5) if ncolumns > 1 else None

        table_hrule_width = 1 if nindex == 1 else 2
        table_vrule_width = 1 if ncolumns == 1 else (1.5 if ncolumns == 2 else 2)

        # padding
        nr, nc = df.shape
        if 'padding_trbl' in kwargs:
            padding_trbl = kwargs['padding_trbl']
        else:
            pad_tb = 4 if nr < 16 else (2 if nr < 25 else 1)
            pad_lr = 10 if nc < 9 else (5 if nc < 13 else 2)
            padding_trbl = (pad_tb, pad_lr, pad_tb, pad_lr)

        font_body = 0.9 if nr < 25 else (0.8 if nr < 41 else 0.7)
        font_caption = np.round(1.1 * font_body, 2)
        font_head = np.round(1.1 * font_body, 2)

        pef_lower = -3
        pef_upper = 6
        pef_precision = 3

        defaults = {
            'ratio_cols': ratio_cols,
            'year_cols': year_cols,
            'default_integer_str': '{x:,.0f}',
            'default_float_str': '{x:,.3f}',
            'default_date_str': '%Y-%m-%d',
            'default_ratio_str': '{x:.1%}',
            'cast_to_floats': True,
            'table_hrule_width': table_hrule_width,
            'table_vrule_width': table_vrule_width,
            'hrule_widths': hrule_widths,
            'vrule_widths': vrule_widths,
            'sparsify': True,
            'sparsify_columns': True,
            'padding_trbl': padding_trbl,
            'font_body': font_body,
            'font_head': font_head,
            'font_caption': font_caption,
            'pef_precision': pef_precision,
            'pef_lower': pef_lower,
            'pef_upper': pef_upper,
            'debug': False
        }
        defaults.update(kwargs)
        super().__init__(df, caption=caption, **defaults)

    @staticmethod
    def guess_years(df):
        """Try to guess which columns (body or index) are years.

        A column is considered a year if:
        - It is numeric (integer or convertible to integer)
        - All values are within a reasonable range (e.g., 1800–2100)
        """
        year_columns = []
        df = df.reset_index(drop=False, col_level=df.columns.nlevels - 1)
        for i, col in enumerate(df.columns):
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if series.dtype.kind in 'iu' and series.between(1800, 2100).all():
                    year_columns.append(col)
            except Exception:
                continue
        return year_columns