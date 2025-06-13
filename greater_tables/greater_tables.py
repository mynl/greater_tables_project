"""Original version 2.0.0."""

# table formatting again
from collections import namedtuple
from decimal import InvalidOperation
from enum import IntEnum
from io import StringIO
from itertools import groupby
import logging
from pathlib import Path
import re
import sys
from textwrap import wrap
import warnings

from bs4 import BeautifulSoup
from cachetools import LRUCache
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, \
    is_float_dtype   # , is_numeric_dtype
from rich import box
from rich.table import Table

from . hasher import df_short_hash
from . gtformats import GT_Format, TableFormat

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
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s |  %(funcName)-15s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(f'Logger Setup; {__name__} module recompiled.')

# TODO Remove!
# temp = None


class Breakability(IntEnum):
    """To track if a column should or should not be broken (wrapped)."""

    NEVER = 0
    DATE = 3
    MAYBE = 5
    ACCEPTABLE = 10



class GT(object):
    """
    Create a greater_tables formatting object.

    Provides html and latex output in quarto/Jupyter accessible manner.
    Wraps AND COPIES the dataframe df. WILL NOT REFLECT CHANGES TO DF.

    Recommended usage is to subclass GT (or use functools.partial) and set
    defaults suitable to your particular
    application. In that way you can maintain a "house-style"

    Process
    --------

    **Input transformation**

    * ``pd.Series`` converted to ``DataFrame``
    * ``list`` converted to  ``DataFrame``, optionally using row 0 as
      ``header_row``
    * A string is  assumed to be a pipe-separated markdown table which is
      converted to a ``DataFrame`` setting aligners per the alignment row
    * All other input types are an error

    The input ``df`` must have unique column names. It is then copied into
    ``self.df`` which will be changed and ``self.raw_df`` for reference.
    The copy is hashed for the table name.

    **Mangling**

    * If show_index, the index is reset and kept, so that all columns are on an
      equal footing
    * The index change levels are computed to determine LaTeX hrules
    * ratio year, and raw columns converted to a list (can be input as a single
      string name)
    * Columns, except raw columns, are cast to floats
    * Column types by index determined
    * default formatter function set (wrapping input, if any)
    * Aligner column input decoded into aligner values
      (``grt-left,grt-right,grt-center``); index aligners separated
    * Formatters decoded, strings mapped to lambda functions as f-string
      formatters, integers as number of decimals
    * Tab values expanded into an iterable
    * Dataframe at this point (index reset, cast) saved to
      ``df_pre_applying_formatters``
    * Determine formatters (``df_formatters`` property, a list of column index
      formatting functions:
        * Make the default float formatter if entered (callable, string, number;
          wrapped in try/except)
        * Determine each column's format type and add function
    * Run ``apply_formatters`` to apply all format choices to ``df``. This
      function handles index columns slightly differently, but results in the
      formatters being applied to each column.
    * Sparsify if requested and if multiindex
    * Result is a dataframe with all object column types and values that
      reflect the formatting choices.


    Parameters
    -----------

    :param df: target DataFrame or list of lists or markdown table string
    :param caption: table caption, optional (GT will look for gt_caption
      attribute of df and use that)
    :param label: TeX label (used in \\label{} command). For markdown
      tables with #tbl:... in the caption it is extracted automatically.
    :param aligners: None or dict (type or colname) -> left | center |
      right
    :param formatters: None or dict (type or colname) -> format function
      for the column; formatters trump ratio_cols
    :param unbreakable: None or list of columns to be considered unbreakable
    :param ratio_cols: None, or "all" or list of column names treated as
      ratios. Set defaults in derived class suitable to application.
    :param year_cols: None, or "all" or list of column names treated as
      years (no commas, no decimals). Set defaults in derived class suitable
      to application.
    :param date_cols: None, or "all" or list of column names treated as
      dates. Set defaults in derived class suitable to application.
    :param raw_cols: None, or "all" or list of column names that are NOT
      cast to floats. Set defaults in derived class suitable to application.
    :param show_index: if True, show the index columns, default True
    :param default_integer_str: format f-string for integers, default
      value '{x:,d}'
    :param default_float_str: format f-string for floats, default
      value '{x:,.3f}'
    :param default_date_str: format f-string for dates, default '%Y-%m-%d'.
      NOTE: no braces or x!
    :param default_ratio_str: format f-string for ratios, default '{x:.1%}'
    :param table_float_format: None or format string for floats in the
      table format function, applied to entire table, default None
    :param table_hrule_width: width of the table top, botton and header
      hrule, default 1
    :param table_vrule_width: width of the table vrule, separating the
      index from the body, default 1
    :param hrule_widths: None or tuple of three ints for hrule widths
      (for use with multiindexes)
    :param vrule_widths: None or tuple of three ints for vrule widths
      (for use when columns have multiindexes)
    :param sparsify: if True, sparsify the index columns, you almost always
      want this to be true!
    :param sparsify_columns: if True, sparsify the columns, default True,
      generally a better look, headings centered in colspans
    :param spacing: 'tight', 'medium', 'wide' to quickly set cell padding.
      Medium is default (2, 10, 2, 10).
    :param padding_trbl: None or tuple of four ints for padding, in order
      top, right, bottom, left.
    :param tikz_scale: scale factor applied to tikz LaTeX tables.
    :param font_body: font size for body text, default 0.9. Units in em.
    :param font_head: font size for header text, default 1.0. Units in em.
    :param font_caption: font size for caption text, default 1.1.
      Units in em.
    :param font_bold_index: if True, make the index columns bold,
      default False.
    :param pef_precision: precision (digits after period) for pandas
      engineering format, default 3.
    :param pef_lower: apply engineering format to floats with absolute
      value < 10**pef_lower; default -3.
    :param pef_upper: apply engineering format to floats with absolute
      value > 10**pef_upper; default 6.
    :param cast_to_floats: if True, try to cast all non-integer, non-date
      columns to floats
    :param header_row: True: use first row as headers; False no headings.
      Default True
    :param tabs: None or list of column widths in characters or a common
      int or float width. (It is converted into em; one character is about
      0.5em on average; digits are exactly 0.5em.) If None, will be calculated.
      Default None.
    :param equal: if True, set all column widths equal. Default False. Maybe
      ignored, depending on computed ideal column widths.
    :param caption_align: for the caption
    :param large_ok: signal that you are intentionally applying to a large
      dataframe. Sub-classes may restrict or apply .head() to df.
    :param max_str_length: maximum displayed length of object types, that
      are cast to strings. Eg if you have nested DataFrames!
    :param str_table_fmt: table border format used for string output
      (markdown), default mixed_grid DEPRECATED??
    :param table_width_mode:
        'explicit': set using max_table_width
        'natural': each cell on one line (can be very wide with long strings)
        'breakable': wrap breakable cells (text strings) at word boundaries
          to fit longest word
        'minimum': wrap breakable and ok-to-break (dates) cells
    :param table_width_header_adjust: additional proportion of table width
      used to balance header columns.
    :param table_width_header_relax: extra spaces allowed per column heading
      to facilitate better column header wrapping.
    :param max_table_width: max table width used for markdown string output,
      default 200; width is never less than minimum width. Padding (3 chars
      per row plus 1) consumed out of max_table_width in string output mode.
    :param debug: if True, add id to caption and use colored lines in table,
      default False.
    """

    # TeX control sequence display widths (heuristic)
    TEX_SIMPLE_GLYPHS = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
        'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'infty',
        'sum', 'prod', 'int', 'cup', 'cap', 'vee', 'wedge', 'forall', 'exists',
        'neg', 'leq', 'geq', 'neq', 'approx', 'to', 'leftarrow', 'rightarrow'
    }
    TEX_WIDE = {'frac', 'sqrt', 'sum', 'int', 'prod'}
    TEX_SPACING = {'quad', 'qquad', ',', ';', ' ', '!'}

    def __init__(self,
                 df,
                 caption='',
                 label='',
                 aligners=None,
                 formatters=None,
                 unbreakable=None,
                 ratio_cols=None,
                 year_cols=None,
                 date_cols=None,
                 raw_cols=None,
                 show_index=True,
                 default_integer_str='{x:,d}',
                 default_float_str='{x:,.3f}',
                 default_date_str='%Y-%m-%d',
                 default_ratio_str='{x:.1%}',
                 default_formatter=None,
                 table_float_format=None,
                 table_hrule_width=1,
                 table_vrule_width=1,
                 hrule_widths=None,
                 vrule_widths=None,
                 sparsify=True,             # index sparsification - almost certainly want this!
                 sparsify_columns=True,     # column sparsification with colspans
                 spacing='medium',          # tight, medium, wide
                 padding_trbl=None,         # tuple of four ints for padding
                 tikz_scale=1.0,
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
                 caption_align='center',
                 large_ok=False,
                 max_str_length=-1,
                 str_table_fmt='mixed_grid',
                 table_width_mode='explicit',
                 table_width_header_adjust=0.1,
                 table_width_header_relax=10,
                 max_table_width=200,
                 debug=False):

        # deal with alternative input modes
        if df is None:
            # don't want None to fail
            df = pd.DataFrame([])
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
            df = df.strip()
            if df == '':
                df = pd.DataFrame([])
            else:
                df, aligners, caption, label = GT.md_to_df(df)
                show_index = False
        else:
            raise ValueError(
                'df must be a DataFrame, a list of lists, or a markdown table string')

        if len(df) > 50 and not large_ok:
            raise ValueError(
                'Large dataframe (>50 rows) and large_ok not set to true...do you know what you are doing?')

        if not df.columns.is_unique:
            raise ValueError('df column names are not unique')

        # extract value BEFORE copying, copying does not carry these attributes over
        if caption != '':
            self.caption = caption
        else:
            # used by querex etc.
            self.caption = getattr(df, 'gt_caption', '')
        self.label = label
        self.df = df.copy(deep=True)   # the object being formatted
        self.raw_df = df.copy(deep=True)
        # if not column_names:
        # get rid of column names
        # self.df.columns.names = [None] * self.df.columns.nlevels
        self.df_id = df_short_hash(self.df)
        # TODO: update / change
        self.str_table_fmt = str_table_fmt
        # TODO: implement
        table_width_mode = table_width_mode.lower()
        if table_width_mode not in ('explicit', 'natural', 'breakable', 'minimum'):
            raise ValueError(f'Inadmissible options {table_width_mode} for table_width_mode.')
        self.table_width_mode = table_width_mode
        self.table_width_header_adjust = table_width_header_adjust
        self.table_width_header_relax = table_width_header_relax
        self.max_table_width = max_table_width
        self.debug = debug
        if self.caption != '' and self.debug:
            self.caption += f' (id: {self.df_id})'
        self.max_str_length = max_str_length
        # before messing
        self.show_index = show_index
        self.nindex = self.df.index.nlevels if self.show_index else 0
        self.ncolumns = self.df.columns.nlevels
        self.ncols = self.df.shape[1]
        self.dt = self.df.dtypes

        # reset index to put all columns on an equal footing, but note number ofindex cols
        with warnings.catch_warnings():
            if self.show_index:
                warnings.simplefilter(
                    "ignore", category=pd.errors.PerformanceWarning)
                self.df = self.df.reset_index(
                    drop=False, col_level=self.df.columns.nlevels - 1)
            # want the new index to be ints - that is not default if old was multiindex
            self.df.index = np.arange(self.df.shape[0], dtype=int)
        self.index_change_level = GT.changed_column(
            self.df.iloc[:, :self.nindex])
        if self.ncolumns > 1:
            # will be empty rows above the index headers
            self.index_change_level = pd.Series(
                [i[-1] for i in self.index_change_level])

        self.column_change_level = GT.changed_level(self.raw_df.columns)

        # determine ratio columns
        if ratio_cols is not None and not self.df.columns.is_unique:
            logger.warning(
                'Ratio cols specified with non-unique column names: ignoring request.')
            self.ratio_cols = []
        else:
            if ratio_cols is None:
                self.ratio_cols = []
            elif ratio_cols == 'all':
                self.ratio_cols = [i for i in self.df.columns]
            elif ratio_cols is not None and not isinstance(ratio_cols, (tuple, list)):
                self.ratio_cols = self.cols_from_regex(
                    ratio_cols)  # [ratio_cols]
            else:
                self.ratio_cols = ratio_cols

        # determine year columns
        if year_cols is not None and not self.df.columns.is_unique:
            logger.warning(
                'Year cols specified with non-unique column names: ignoring request.')
            self.year_cols = []
        else:
            if year_cols is None:
                self.year_cols = []
            elif year_cols is not None and not isinstance(year_cols, (tuple, list)):
                self.year_cols = self.cols_from_regex(year_cols)  # [year_cols]
            else:
                self.year_cols = year_cols

        # determine date columns
        if date_cols is not None and not self.df.columns.is_unique:
            logger.warning(
                'Year cols specified with non-unique column names: ignoring request.')
            self.date_cols = []
        else:
            if date_cols is None:
                self.date_cols = []
            elif date_cols is not None and not isinstance(date_cols, (tuple, list)):
                self.date_cols = self.cols_from_regex(date_cols)  # [date_cols]
            else:
                self.date_cols = date_cols

        # determine columns NOT to cast to floats
        if raw_cols is not None and not self.df.columns.is_unique:
            logger.warning(
                'Year cols specified with non-unique column names: ignoring request.')
            self.raw_cols = []
        else:
            if raw_cols is None:
                self.raw_cols = []
            elif raw_cols is not None and not isinstance(raw_cols, (tuple, list)):
                self.raw_cols = self.cols_from_regex(raw_cols)  # [raw_cols]
            else:
                self.raw_cols = raw_cols

        # figure the default formatter (used in conjunction with raw columns)
        if default_formatter is None:
            self.default_formatter = self._default_formatter
        else:
            assert callable(
                default_formatter), 'default_formatter must be callable'

            def wrapped_default_formatter(x):
                try:
                    return default_formatter(x)
                except ValueError:
                    return str(x)
            self.default_formatter = wrapped_default_formatter

        # cast as much as possible to floats
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=pd.errors.PerformanceWarning)
            if cast_to_floats:
                for i, c in enumerate(self.df.columns):
                    if c in self.raw_cols or c in self.date_cols:
                        continue
                    old_type = self.df.dtypes[c]
                    if not np.any((is_integer_dtype(self.df.iloc[:, i]),
                                   is_datetime64_any_dtype(self.df.iloc[:, i]))):
                        try:
                            self.df.iloc[:, i] = self.df.iloc[:,
                                                              i].astype(float)
                            logger.debug(
                                f'coerce {i}={c} from {old_type} to float')
                        except (ValueError, TypeError):
                            logger.debug(
                                f'coercing {i}={c} from {old_type} to float FAILED')

        # massage unbreakable
        if unbreakable is None:
            unbreakable = []
        elif isinstance(unbreakable, str):
            unbreakable = [unbreakable]

        # now can determine types and infer the break penalties (for column sizes)
        self.float_col_indices = []
        self.integer_col_indices = []
        self.date_col_indices = []
        self.object_col_indices = []  # not actually used, but for neatness
        self.break_penalties = []
        # manage non-unique col names here
        logger.debug('FIGURING TYPES')
        for i, cn in enumerate(self.df.columns):  # range(self.df.shape[1]):
            ser = self.df.iloc[:, i]
            if cn in self.date_cols:
                logger.debug(f'col {i}/{cn} specified as date col')
                self.date_col_indices.append(i)
                self.break_penalties.append(
                    Breakability.NEVER if cn in unbreakable else Breakability.DATE)
            elif is_datetime64_any_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is DATE')
                self.date_col_indices.append(i)
                self.break_penalties.append(
                    Breakability.NEVER if cn in unbreakable else Breakability.DATE)
            elif is_integer_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is INTEGER')
                self.integer_col_indices.append(i)
                self.break_penalties.append(
                    Breakability.NEVER if cn in unbreakable else Breakability.NEVER)
            elif is_float_dtype(ser):
                logger.debug(f'col {i} = {self.df.columns[i]} is FLOAT')
                self.float_col_indices.append(i)
                self.break_penalties.append(
                    Breakability.NEVER if cn in unbreakable else Breakability.NEVER)
            else:
                logger.debug(f'col {i} = {self.df.columns[i]} is OBJECT')
                self.object_col_indices.append(i)
                c = ser.name
                if c in self.year_cols or c in self.ratio_cols:
                    self.break_penalties.append(
                        Breakability.NEVER if cn in unbreakable else Breakability.NEVER)
                else:
                    self.break_penalties.append(
                        Breakability.NEVER if cn in unbreakable else Breakability.ACCEPTABLE)

        # figure out column and index alignment
        if aligners is not None and np.any(self.df.columns.duplicated()):
            logger.warning(
                'aligners specified with non-unique column names: ignoring request.')
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
            elif c in self.year_cols:
                self.df_aligners.append('grt-center')
            elif c in self.raw_cols:
                # these are strings
                self.df_aligners.append('grt-left')
            elif i in self.date_col_indices:
                # center dates, why not!
                self.df_aligners.append('grt-center')
            elif c in self.ratio_cols or i in self.float_col_indices or i in self.integer_col_indices:
                # number -> right
                self.df_aligners.append('grt-right')
            else:
                # all else, left
                self.df_aligners.append('grt-left')

        self.df_idx_aligners = self.df_aligners[:self.nindex]

        if formatters is None:
            self.default_formatters = {}
        else:
            self.default_formatters = {}
            for k, v in formatters.items():
                if callable(v):
                    self.default_formatters[k] = v
                elif type(v) == str:
                    self.default_formatters[k] = lambda x: v.format(x=x)
                elif type(v) == int:
                    fmt = f'{{x:.{v}f}}'
                    self.default_formatters[k] = lambda x: fmt.format(x=x)
                else:
                    raise ValueError(
                        'formatters must be dict of callables or ints or format strings {x:...}')

        # store defaults
        self.default_integer_str = default_integer_str
        # VERY rarely used; for floats in cols that are not floats
        self.default_float_str = default_float_str
        self.default_date_str = default_date_str.replace(
            '{x:', '').replace('}', '')
        self.default_ratio_str = default_ratio_str
        self.pef_precision = pef_precision
        self.pef_lower = pef_lower
        self.pef_upper = pef_upper
        self._pef = None
        self.table_float_format = table_float_format
        self.default_float_formatter = None
        self.hrule_widths = hrule_widths or (0, 0, 0)
        if not isinstance(self.hrule_widths, (list, tuple)):
            self.hrule_widths = (self.hrule_widths,)
        self.vrule_widths = vrule_widths or (0, 0, 0)
        if not isinstance(self.hrule_widths, (list, tuple)):
            self.hrule_widths = (self.hrule_widths, )
        self.table_hrule_width = table_hrule_width
        self.table_vrule_width = table_vrule_width
        self.font_body = font_body
        self.font_head = font_head
        self.font_caption = font_caption
        self.tikz_scale = tikz_scale
        self.font_bold_index = font_bold_index
        self.caption_align = caption_align
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
                raise ValueError(
                    'spacing must be tight, medium, or wide or tuple of four ints.')
        try:
            self.padt, self.padr, self.padb, self.padl = padding_trbl
        except ValueError:
            logger.error(
                f'padding_trbl {padding_trbl=}, must be four ints, defaultign to medium')
            self.padt, self.padr, self.padb, self.padl = 2, 10, 2, 10

        # because of the problem of non-unique indexes use a list and
        # not a dict to pass the formatters to to_html
        self._df_formatters = None
        self.df_style = ''
        self.df_html = ''
        self._clean_html = ''
        self._clean_tex = ''
        self._rich_table = None
        # finally sparsify and then apply formaters
        # this radically alters the df, so keep a copy for now...
        self.df_pre_applying_formatters = self.df.copy()
        self.df = self.apply_formatters(self.df)
        # cache for various things...
        self._cache = LRUCache(20)
        # sparsify
        if sparsify and self.nindex > 1:
            self.df = GT.sparsify(self.df, self.df.columns[:self.nindex])
            # for c in self.df.columns[:self.nindex]:
            #     # sparsify returns some other stuff...
            #     self.df[c], _ = GT.sparsify(self.df[c])

    def __repr__(self):
        """Basic representation."""
        return f"GreaterTable(df_id={self.df_id})"

    def __str__(self):
        """String representation, for print()."""
        return self.to_string_custom()

    def _repr_html_(self):
        """
        Apply format to self.df.

        ratio cols like in constructor
        """
        return self.html

    def _repr_latex_(self):
        """Generate a LaTeX tabular representation."""
        # return ''
        # latex = self.df.to_latex(caption=self.caption, formatters=self._df_formatters)
        if self._clean_tex == '':
            self._clean_tex = self.make_tikz()
            logger.info('CREATED LATEX')
        return self._clean_tex

    def cols_from_regex(self, regex):
        """Return columns of self.df matching regex"""
        return [col for col in self.df.columns if isinstance(col, str) and re.search(regex, col)]

    def cache_get(self, key):
        """Retrieve item from cache."""
        return self._cache.get(key, None)

    def cache_set(self, key, value):
        """Add item to cache."""
        self._cache[key] = value

    # define the default and easy formatters ===================================================
    def default_ratio_formatter(self, x):
        """Ratio formatter."""
        try:
            return self.default_ratio_str.format(x=x)
        except ValueError:
            return str(x)

    def default_date_formatter(self, x):
        """Date formatter that works for strings too."""
        if pd.isna(x):
            return ""
        try:
            dt = pd.to_datetime(x, errors='coerce')
            if pd.isna(dt):
                return str(x)
            return dt.strftime(self.default_date_str)
        except Exception:
            logger.error("date error with %s", x)
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

    def default_raw_formatter(self, x):
        """Formatter for columns flagged as raw."""
        return str(x)

    # def default_formatter(self, x):
    #     """Universal formatter for other types."""
    #     try:
    #         # werid wrinkle here: float('infinity') -> np.inf!!
    #         f = float(x)
    #         if self.default_float_formatter:
    #             return self.default_float_formatter(f)
    #         try:
    #             i = int(x)
    #         except ValueError:
    #             try:
    #                 i = int(f)
    #             except OverflowError:
    #                 # this came up! Passed the work "Infinity"
    #                 return str(x)
    #         if i == f:
    #             return self.default_integer_str.format(x=i)
    #         else:
    #             # TODO BEEF UP?
    #             return self.default_float_str.format(x=f)
    #     except (TypeError, ValueError):
    #         if self.max_str_length < 0:
    #             return str(x)
    #         else:
    #             return str(x)[:self.max_str_length]

    def _default_formatter(self, x):
        """Default universal formatter for other types (GTP re-write of above cluster."""
        try:
            f = float(x)
        except (TypeError, ValueError):
            s = str(x)
            return s if self.max_str_length < 0 else s[:self.max_str_length]

        if self.default_float_formatter:
            return self.default_float_formatter(f)

        if np.isinf(f) or np.isnan(f):  # clearer handling of weird float cases
            return str(x)

        if f.is_integer():
            return self.default_integer_str.format(x=int(f))
        else:
            return self.default_float_str.format(x=f)

    def pef(self, x):
        """Pandas engineering format."""
        if self._pef is None:
            self._pef = pd.io.formats.format.EngFormatter(accuracy=self.pef_precision, use_eng_prefix=True)   # noqa
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
        if self._df_formatters is None:
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
                        raise ValueError(
                            'table_float_format must be a string or a function')
                    fmt = self.table_float_format

                    def ff(x):
                        try:
                            return fmt.format(x=x)
                        except ValueError:
                            return str(x)
                        except Exception as e:
                            logger.error(
                                f'Custom float format string raised {e=}')
                    self.default_float_formatter = ff
            else:
                self.default_float_formatter = False

            self._df_formatters = []
            for i, c in enumerate(self.df.columns):
                # set a default, note here can have
                # non-unique index so work with position i
                if c in self.default_formatters:
                    self._df_formatters.append(self.default_formatters[c])
                elif c in self.ratio_cols:
                    # print(f'{i} ratio')
                    self._df_formatters.append(self.default_ratio_formatter)
                elif c in self.year_cols:
                    self._df_formatters.append(self.default_year_formatter)
                elif c in self.raw_cols:
                    self._df_formatters.append(self.default_raw_formatter)
                elif i in self.date_col_indices:
                    self._df_formatters.append(self.default_date_formatter)
                elif i in self.integer_col_indices:
                    # print(f'{i} int')
                    self._df_formatters.append(self.default_integer_formatter)
                elif i in self.float_col_indices:
                    # trickier approach...
                    self._df_formatters.append(
                        self.default_float_formatter or self.make_float_formatter(self.df.iloc[:, i]))
                else:
                    # print(f'{i} default')
                    self._df_formatters.append(self.default_formatter)
            # self._df_formatters is now a list of length equal to cols in df
            if len(self._df_formatters) != self.df.shape[1]:
                raise ValueError(
                    f'Something wrong: {len(self._df_formatters)=} != {self.df.shape=}')
        return self._df_formatters

    def make_column_width_df(self):
        """
        Return dataframe of width information.

        Returned dataframe has columns for

        * natural width, all on one line = max len by col
        * min width = max length given breaks
        * break type of column
        * alignment of column
        * index natural width
        * index min width
        """
        df = self.df
        n_row, n_col = df.shape

        # The width if content didn't wrap (single line)
        # Series=dict colname->max width of cells in column
        natural_width = df.map(lambda x: len(x.strip())).max(axis=0).to_dict()

        # re.split(r'(?<=[\s.,:;!?()\[\]{}\-\\/|])\s*', text)
        # (?<=...) is a lookbehind to preserve the break character with the left-hand fragment.
        # [\s.,:;!?()\[\]{}\-\\/|] matches common punctuation and separators:
        # \s = whitespace
        # . , : ; ! ? = terminal punctuation
        # () [] {} = brackets
        # \- = dash
        # \\/| = slash, backslash, pipe
        pat = r'(?<=[.,;:!?)\]}\u2014\u2013])\s+|--+\s+|\s+'
        iso_date_split = r'(?<=\b\d{4})-(?=\d{2}-\d{2})'
        pat = f'{pat}|{iso_date_split}'

        # Calculate ideal (no wrap) and minimum possible widths for all columns
        # The absolute minimum width each column can take (e.g., longest word for text)
        min_acceptable_width = {}
        for col_name in df.columns:
            min_acceptable_width[col_name] = (
                df[col_name].str
                .split(pat=pat, regex=True, expand=True)
                .fillna('')
                .map(len)
                .max(axis=1)
                .max()
            )
        # ans will be the col_width_df
        ans = pd.DataFrame({
            'alignment': [i[4:] for i in self.df_aligners],
            'break_penalties': self.break_penalties,
            'breakability': [x.name for x in self.break_penalties],
            'natural_width': natural_width.values(),
            'min_acceptable_width': min_acceptable_width.values(),
        }, index=df.columns)
        ans['break_acceptable'] = np.where(
            ans.break_penalties == Breakability.ACCEPTABLE, ans.min_acceptable_width, ans.natural_width)
        # DUH - this is min_acceptable_width
        # ans['break_dates'] = np.where(ans.break_penalties==Breakability.DATE, ans.min_acceptable_width, ans.break_acceptable)

        natural, acceptable, min_acceptable = ans.iloc[:, 3:].sum()
        PADDING = 2  # per column
        if self.table_width_mode == 'explicit':
            # target width INCLUDES padding and column marks |
            target_width = self.max_table_width - (PADDING + 1) * n_col - 1
            logger.info(f'Col padding effect {self.max_table_width=} ==> {target_width=}')
        elif self.table_width_mode == 'natural':
            target_width = natural + (PADDING + 1) * n_col + 1
        elif self.table_width_mode == 'breakable':
            target_width = acceptable + (PADDING + 1) * n_col + 1
        elif self.table_width_mode == 'minimum':
            target_width = min_acceptable + (PADDING + 1) * n_col + 1

        # extra space for the headers to relax, if useful
        if self.table_width_header_adjust > 0:
            max_extra = int(self.table_width_header_adjust * target_width)
        else:
            max_extra = 0

        if target_width > natural:
            # everything gets its natural width
            ans['recommended'] = ans['natural_width']
            space = target_width - natural
            logger.info('Space for NATURAL! Spare space = %s', space)
        elif target_width > acceptable:
            # strings wrap
            ans['recommended'] = ans['break_acceptable']
            # use up extra on the ACCEPTABLE cols
            space = target_width - acceptable
            logger.info(
                'Using breaks acceptable (dates not wrapped), spare space = %s', space)
        elif target_width > min_acceptable:
            # strings and dates wrap
            ans['recommended'] = ans['min_acceptable_width']
            # use up extra on dates first, then strings
            space = target_width - min_acceptable
            logger.info(
                'Breaking all breakable (incl dates), spare space = %s', space)
        else:
            # OK severely too small
            ans['recommended'] = ans['min_acceptable_width']
            logger.info(
                'Desired width too small for pleasant formatting, table will be too wide.')
            space = target_width - min_acceptable

        input_df = None
        if space >= 0:
            # Allocate the excess ------------------------------
            # Fancy col headings currently only for 1-d index
            # TODO NOTE: use sparsify logic you have for index applied to df.T
            # to sort the columns!!
            if df.columns.nlevels == 1:
                # Step 1: baseline comes in from code above
                ans['raw_rec'] = ans['recommended']

                # Step 2: get rid of intra-line breaks
                if max_extra > 0:
                    adj, input_df = self.header_adjustment(
                        df, ans['recommended'], space, max_extra)
                    # create new col and populate per GPT
                    ans['header_tweak'] = pd.Series(adj)
                else:
                    ans['header_tweak'] = 0
                ans['recommended'] = ans['recommended'] + ans['header_tweak']
                ans['natural_w_header'] = ans['recommended']
            else:
                # avoid a failure blow
                ans['raw_rec'] = np.nan
                ans['header_tweak'] = np.nan
                ans['natural_w_header'] = np.nan
            # Step 3: distribute remaining slack proportionally
            remaining = target_width - ans['recommended'].sum()
            if remaining > 0:
                slack = ans['natural_width'] - ans['recommended']
                total_slack = slack.clip(lower=0).sum()
                if total_slack > 0:
                    fractions = slack.clip(lower=0) / total_slack
                    ans['recommended'] += np.floor(fractions *
                                                   remaining).astype(int)
                    ans['recommended'] = np.maximum(
                        ans['recommended'], ans['natural_w_header'])

            # Ensure final constraint
            ans['recommended'] = ans['recommended'].astype(int)
            logger.info("Raw rec: %s\tTweaks: %s\tActual: %s\tTarget: %s\tOver/(U): %s",
                        ans['raw_rec'].sum(),
                        ans['header_tweak'].sum(),
                        ans['recommended'].sum(),
                        target_width,
                        ans['recommended'].sum() - target_width
                        )
            ans = ans[[
                'alignment',
                'break_penalties',
                'breakability',
                'natural_width',
                'break_acceptable',
                'min_acceptable_width',
                'raw_rec',
                'header_tweak',
                'natural_w_header',
                'recommended',
            ]]
        # in all cases...
        # need recommended to be > 0
        ans['recommended'] = np.maximum(ans['recommended'], 1)
        self.cache_set('column_width_df', ans)
        # info about the header adjustment
        self.cache_set('input_df', input_df)

        return ans

    @staticmethod
    def header_adjustment(df, min_widths, space, max_extra):
        """
        Fine-adjust heading for optimal spacing.

        Return a dict with per-column recommended width adjustments to avoid
        intra-word breaks and reduce overall header height.

        Parameters:fv
            df: DataFrame with 1-level string column names
            min_widths: dict of column name -> minimal acceptable width
            space: amount of space available to be allocated
            max_extra: max extra characters to consider allocating per column

        Returns:
            dict: column -> additional width to allocate
        """
        colnames = list(df.columns)
        adjustments = {col: 0 for col in colnames}
        num_lines = 0

        def has_intra_word_break(text: str, width: int) -> bool:
            """
            Determine if textwrap.wrap breaks any words in the given text.

            Gemini - GPT code did not work, even after seveal iterations.
            This is a nice approach to the problem.

            Args:
                text: The input string.
                width: The maximum width for wrapping.

            Returns:
                True if any word is broken across lines, False otherwise.
            """
            nonlocal num_lines
            wrapped_lines = wrap(text, width=width)
            num_lines = len(wrapped_lines)
            original_words = text.split()

            reconstructed_text_from_wrapped = " ".join(wrapped_lines)
            reconstructed_words = reconstructed_text_from_wrapped.split()

            # If the number of words differs, it means some words were split.
            # This catches cases where a word might be split and then later re-joined
            # due to subsequent wrapping logic, leading to a different number of words.
            if len(original_words) != len(reconstructed_words):
                return True

            # Compare word by word. If any word from the original doesn't exactly match
            # a word from the reconstructed list, it implies a split.
            for i in range(len(original_words)):
                if original_words[i] != reconstructed_words[i]:
                    return True

            return False

        # First pass: avoid ugly intraword breaks
        # make dict of col -> longest word length
        min_acceptable = {c: v for c, v in
                          zip(colnames, map(lambda x: max(len(i) for i in re.split(r'[ \-/]', x)), colnames))}
        options = []
        for col in colnames:
            if not isinstance(col, str):
                continue
            base_width = min_widths[col]
            if not has_intra_word_break(col, base_width):
                options.append([col, 0, num_lines])
                # nothing to be gained, move to next col
                continue
            extra0 = max(0, min_acceptable[col] - base_width)
            if extra0 > max_extra:
                # ok, can't flatten word because it is too long
                extra0 = 0
            elif extra0 == max_extra:
                # go with that
                adjustments[col] = max_extra
                continue
            # see if col can be flattened within max_extra chars, starting
            # at extra0, which is enough to avoid intraword breaks
            for extra in range(extra0, max_extra + 1):
                if not has_intra_word_break(col, base_width + extra):
                    options.append([col, extra, num_lines])
                    if adjustments[col] == 0:
                        # take first, but compute rest...
                        adjustments[col] = extra
            # temporary diagnostic DEBUG information - comment in prod
            # from IPython.display import display
            # debug = pd.Series([col, min_acceptable[col], base_width, has_intra_word_break(col, base_width), extra0, max_extra,
            #     wrap(col,  base_width), extra],
            #     index=['col name', 'min acceptable', 'base_width (from data)', 'intra word break', 'extra0', 'max_extra', 'split', 'selected extra']).to_frame('Value')
            # display(debug)
        # make df[col name, amount of extra space for col, resulting number of lines]
        # this is needed as input for the optimal heading function (next)
        input_df = pd.DataFrame(options, columns=['col', 'extra', 'num_lines'])
        # min amount to avoid intra work breaks
        avoid_intra = input_df.groupby('col').min().extra.sum()
        if avoid_intra >= space:
            # that's all we can do
            print("NO FURTHER IMPROVEMENTS")
        else:
            # can try for a better solution
            sol = GT.optimal_heading(input_df, space)
            adjustments.update(sol[1])
            logger.info('best solution: %s', sol)
        # global temp
        # temp = input_df
        return adjustments, input_df

    @staticmethod
    def optimal_heading(input_df: pd.DataFrame, total_es_budget: int) -> tuple[int, dict[str, int]]:
        """
        Optimize extra spacing for best heading.

        Finds the best way to allocate extra space to minimize max_lines in heading.

        Gemini solution.

        Args:
            input_df: DataFrame with 'col', 'extra', 'num_lines'.
            total_es_budget: The total extra space to allocate.

        Returns:
            A tuple: (min_max_lines, optimal_extra_allocation_per_column).

        .. _table_layout_optimization:

        Table Layout Optimization
        =========================

        This document describes the algorithm implemented in the :py:func:`find_best_layout` function, which aims to optimize the allocation of a fixed amount of extra space (`ES`) among table columns to minimize the overall table height (i.e., the maximum number of lines used by any single column).

        Problem Statement
        -----------------

        Given a set of table columns, each with a known relationship between allocated "extra space" and the resulting "number of lines" it occupies when wrapped, and a total budget of extra space, the goal is to find an allocation of this extra space to each column such that the maximum number of lines among all columns is minimized.

        For example, a column named "location category (float)" might take 3 lines with 0 extra space, but perhaps only 2 lines with 2 extra space, and 1 line with 5 extra space. The relationship is provided in a Pandas DataFrame with columns `col`, `extra`, and `num_lines`.

        Algorithm: Binary Search on the Answer
        -------------------------------------

        The problem exhibits a monotonic property: if a table layout can be achieved with a maximum height of `X` lines, it can also be achieved with any maximum height `Y > X` lines (by simply using the same or more `extra` space). This property makes binary search on the *minimum possible maximum lines* an efficient solution.

        The algorithm proceeds as follows:

        1.  **Preprocessing the Input Data:**
            The input `pandas.DataFrame` is processed to create a convenient lookup structure. For each unique column, a sorted list of `(extra_space, num_lines)` tuples is created. This allows for quick identification of the minimum `extra` space required for a given `column` to fit within a `target_max_lines`.

            .. code-block:: python

                unique_cols = input_df['col'].unique().tolist()
                col_extra_num_lines_options = {}
                for col_name in unique_cols:
                    col_data = input_df[input_df['col'] == col_name].sort_values(by='extra')
                    col_extra_num_lines_options[col_name] = list(zip(col_data['extra'], col_data['num_lines']))

        2.  **Defining the Search Space (Bounds for `max_lines`):**
            The binary search operates on the possible values for the `optimal_max_lines`.
            * **Lower Bound (`L`):** The absolute minimum number of lines observed across all columns and all `extra` space options in the input data. This represents the theoretical minimum height a column could ever achieve.
            * **Upper Bound (`R`):** The absolute maximum number of lines observed across all columns and all `extra` space options in the input data. This represents the worst-case height, which is always achievable.

            .. code-block:: python

                all_num_lines = input_df['num_lines'].unique()
                if len(all_num_lines) == 0:
                    return 0, {} # Handle empty DataFrame case
                L = all_num_lines.min()
                R = all_num_lines.max()

        3.  **The `check(target_max_lines)` Function:**
            This is the core helper function for the binary search. Given a `target_max_lines` (a candidate for the overall maximum height), it determines if it's *possible* to achieve this height for *all* columns simultaneously, without exceeding the `total_es_budget`.

            For each column:
            * It iterates through its `(extra_space, num_lines)` options (which are sorted by `extra_space`).
            * It finds the *smallest* `extra_space` value for which the corresponding `num_lines` is less than or equal to `target_max_lines`.
            * If no such `extra_space` is found for a column (meaning even with the maximum available `extra` for that column, it still exceeds `target_max_lines`), then `target_max_lines` is not achievable, and the function returns `False`.
            * Otherwise, it sums up these minimum required `extra_space` values across all columns.
            * If the total `extra_space` required is less than or equal to `total_es_budget`, the function returns `True` (meaning `target_max_lines` is achievable). Otherwise, it returns `False`.

            .. code-block:: python

                def check(target_max_lines: int) -> bool:
                    current_extra_needed = 0
                    for col_name in unique_cols:
                        min_extra_for_col = float('inf')
                        found_suitable_extra = False
                        for extra_val, num_lines_val in col_extra_num_lines_options[col_name]:
                            if num_lines_val <= target_max_lines:
                                min_extra_for_col = extra_val
                                found_suitable_extra = True
                                break # Found the minimum extra for this column

                        if not found_suitable_extra:
                            return False # This target_max_lines is too low for this column

                        current_extra_needed += min_extra_for_col

                    return current_extra_needed <= total_es_budget

        4.  **Binary Search Loop:**
            The main binary search loop iteratively narrows down the range `[L, R]`.
            * In each iteration, it calculates the `mid_max_lines = L + (R - L) // 2`.
            * It then calls the `check(mid_max_lines)` function.
            * If `check(mid_max_lines)` returns `True` (meaning `mid_max_lines` is achievable):
                * `mid_max_lines` becomes a candidate for the `optimal_max_lines`. We record the current allocation that achieved it.
                * We try to achieve an even smaller `max_lines` by setting `R = mid_max_lines - 1`.
            * If `check(mid_max_lines)` returns `False` (meaning `mid_max_lines` is not achievable):
                * We need to allow for more lines, so we set `L = mid_max_lines + 1`.

            The loop continues until `L > R`, at which point `optimal_max_lines` will hold the smallest possible maximum height, and `best_allocation` will store the corresponding `extra_space` allocation for each column.

            .. code-block:: python

                optimal_max_lines = R
                best_allocation = {}

                while L <= R:
                    mid_max_lines = L + (R - L) // 2

                    # Recalculate allocation within the loop to store the specific 'extra' values
                    temp_current_extra_needed = 0
                    temp_current_allocation = {}
                    possible = True
                    for col_name in unique_cols:
                        min_extra_for_col = float('inf')
                        found_suitable_extra = False
                        for extra_val, num_lines_val in col_extra_num_lines_options[col_name]:
                            if num_lines_val <= mid_max_lines:
                                min_extra_for_col = extra_val
                                found_suitable_extra = True
                                break

                        if not found_suitable_extra:
                            possible = False
                            break

                        temp_current_extra_needed += min_extra_for_col
                        temp_current_allocation[col_name] = min_extra_for_col

                    if possible and temp_current_extra_needed <= total_es_budget:
                        optimal_max_lines = mid_max_lines
                        best_allocation = temp_current_allocation.copy()
                        R = mid_max_lines - 1
                    else:
                        L = mid_max_lines + 1

            The function returns the `optimal_max_lines` and the `best_allocation` dictionary, mapping each column name to the minimal `extra_space` it needs to achieve that optimal height.

        Why this approach is effective:
        ------------------------------

        * **Optimal Solution:** The binary search guarantees finding the absolute minimum possible `max_lines` because it systematically explores the entire solution space.
        * **Efficiency:** The `check` function runs in time proportional to the number of columns times the average number of `extra` options per column. The binary search itself performs `log(range_of_num_lines)` iterations. This makes the overall complexity efficient for typical table sizes.
        * **Flexibility:** It does not assume any particular mathematical function relating `extra` space to `num_lines`. It works with arbitrary discrete relationships provided in the input DataFrame, as long as `num_lines` is non-increasing as `extra` increases (which is the natural expectation for this problem).


        """
        # Pre-processing
        unique_cols = input_df['col'].unique().tolist()

        col_extra_num_lines_options = {}
        for col_name in unique_cols:
            col_data = input_df[input_df['col'] ==
                                col_name].sort_values(by='extra')
            col_extra_num_lines_options[col_name] = list(
                zip(col_data['extra'], col_data['num_lines']))

        def check(target_max_lines: int) -> bool:
            current_extra_needed = 0
            for col_name in unique_cols:
                min_extra_for_col = float('inf')
                found_suitable_extra = False
                for extra_val, num_lines_val in col_extra_num_lines_options[col_name]:
                    if num_lines_val <= target_max_lines:
                        min_extra_for_col = extra_val
                        found_suitable_extra = True
                        break

                if not found_suitable_extra:
                    return False

                current_extra_needed += min_extra_for_col

            return current_extra_needed <= total_es_budget

        all_num_lines = input_df['num_lines'].unique()

        # Corrected line: Check length of the numpy array
        if len(all_num_lines) == 0:
            return 0, {}

        L = all_num_lines.min()
        R = all_num_lines.max()

        optimal_max_lines = R
        best_allocation = {}

        while L <= R:
            mid_max_lines = L + (R - L) // 2

            temp_current_extra_needed = 0
            temp_current_allocation = {}
            possible = True
            for col_name in unique_cols:
                min_extra_for_col = float('inf')
                found_suitable_extra = False
                for extra_val, num_lines_val in col_extra_num_lines_options[col_name]:
                    if num_lines_val <= mid_max_lines:
                        min_extra_for_col = extra_val
                        found_suitable_extra = True
                        break

                if not found_suitable_extra:
                    possible = False
                    break

                temp_current_extra_needed += min_extra_for_col
                temp_current_allocation[col_name] = min_extra_for_col

            if possible and temp_current_extra_needed <= total_es_budget:
                optimal_max_lines = mid_max_lines
                best_allocation = temp_current_allocation.copy()
                R = mid_max_lines - 1
            else:
                L = mid_max_lines + 1

        return optimal_max_lines, best_allocation

    def to_string_custom(self):
        """Print to string using new functionality."""
        if self.df.empty:
            return ""

        cw_df = self.make_column_width_df()
        cw = cw_df['recommended']
        aligners = cw_df['alignment']
        txt = GT.to_text_table(self.df, cw, aligners, index_levels=self.nindex)
        return txt

    def to_string_tabulate(self):
        """(Old) string representation using tabulate but with new col widther."""
        if self.df.empty:
            return ""

        cw_df = self.make_column_width_df()
        cw = list(cw_df['recommended'])
        aligners = list(cw_df['alignment'])
        txt = self.df.to_markdown(
            index=False,  # NEVER show index; it's subsumed into self.df
            colalign=aligners,
            tablefmt=self.str_table_fmt,
            maxcolwidths=cw,
            maxheadercolwidths=cw,
        )
        return txt

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
    /* center tables in quarto context */
    .greater-table {{
        display: block;
        text-align: center;
    }}
    .greater-table > table {{
        display: inline-table;
    }}
    /* tag formats */
    #{self.df_id} caption {{
        padding: {2 * padt}px {padr}px {padb}px {padl}px;
        font-size: {self.font_caption}em;
        text-align: {self.caption_align};
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
        index_name_to_level = dict(
            zip(self.raw_df.index.names, range(self.nindex)))
        index_change_level = self.index_change_level.map(index_name_to_level)
        # this is easier and computed in the init
        column_change_level = self.column_change_level

        # Start table
        html = [f'<table id="{self.df_id}">']
        if self.label != "":
            pass
            # TODO put in achor tag somehow!!
        if self.caption != '':
            html.append(f'<caption>{self.caption}</caption>')

        # Process header: allow_duplicates=True means can create cols with the same name
        bit = self.df.T.reset_index(drop=False, allow_duplicates=True)
        idx_header = bit.iloc[:self.nindex, :self.ncolumns]
        columns = bit.iloc[self.nindex:, :self.ncolumns]

        colw, tabs = GT.estimate_column_widths(
            self.df, self.max_table_width, nc_index=self.nindex, scale=1, equal=self.equal)
        if self.debug:
            print(f'Make html Input {self.tabs=}\nComputed {tabs=}')
        if self.tabs is not None:
            if len(tabs) == len(self.tabs):
                tabs = self.tabs
            elif len(self.tabs) == 1:
                tabs = self.tabs * len(tabs)
            else:
                logger.error(
                    f'{self.tabs=} must be None, a single number, or a list of numbers of the correct length. Ignoring.')
        # print('HTML ' + ', '.join([f'{c:,.2f}' for c in tabs]))

        # set column widths; tabs returns lengths of strings in each column
        # for proportional fonts, average char is 0.4 to 0.5 em but numbers with
        # tabular-nums are fixed 0.5, so use that
        # scale: want tables about 150-200 char wide, 1 char = 0.5 px size of font
        # so what 75-100 em wide in total
        # add the padding
        # TODO FONT SIZE
        # /4 works well for the tests (handles dates) but seems a bit illogical...
        # guessing font size...
        tabs = np.array(tabs) + (self.padl + self.padr) / 12
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
                for j, (nm, g) in enumerate(groupby(columns.iloc[:, :i + 1].
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
                        html.append(
                            f'<th colspan="{colspan}" class="grt-left {hrule} {vrule}">{nm}</th>')
                    else:
                        html.append(
                            f'<th colspan="{colspan}" class="grt-center {hrule} {vrule}">{nm}</th>')
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
                    html.append(
                        f'<th class="grt-center {hrule} {vrule}">{r}</th>')
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
                    # here have to be careful - if the index is not ! then not every row
                    # appears in the index change level. But if it DOES NOT appear then
                    # it isn't a change level so no rule required
                    if i > 0 and hrule == '' and i in index_change_level and j == index_change_level[i]:
                        hrule = f'grt-hrule-{j}'
                    # html.append(f'<td class="grt-dx-r-{i} grt-dx-c-{j} {self.df_aligners[j]} {hrule}">{c}</td>')
                    col_id = f'grt-c-{j}'
                    html.append(
                        f'<td class="{col_id} {bold_idx} {self.df_aligners[j]} {hrule}">{c}</td>')
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
                html.append(
                    f'<td class="{col_id} {self.df_aligners[j+self.nindex]} {hrule} {vrule}">{c}</td>')
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
                    cleaned_css = re.sub(
                        r'/\*.*?\*/', '', style_tag.string, flags=re.DOTALL)
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
            self._clean_html = str(soup)  # .prettify() -> too many newlines
            logger.info('CREATED HTML')
        return self._clean_html

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
        # Convert MultiIndex to a DataFrame
        index_df = idx.to_frame(index=False)
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
            df_index = df.reset_index(
                drop=False, col_level=self.df.columns.nlevels - 1).iloc[:, :self.nindex]
            new_index = GT.apply_formatters_work(df_index, index_formatters)
            # put them back together
            new_df = pd.concat([new_index, new_body], axis=1)
            new_df = new_df.set_index(list(df_index.columns))
            new_df.index.names = df.index.names
            return new_df
        else:
            raise ValueError(f'unknown mode {mode}')

    def make_tikz(self,
                  column_sep=4 / 8,   # was 3/8
                  row_sep=1 / 8,
                  container_env='table',
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

        * hrule at i means below row i of the table. (1-based) Top, bottom and
          below index lines are inserted automatically. Top and bottom lines
          are thicker.
        * vrule at i means to the left of table column i (1-based); there will
          never be a rule to the far right...it looks plebby; remember you must
          include the index columns!

        sparsify  number of cols of multi index to sparsify

        Issue: column with floats and spaces or missing causes problems (VaR,
        TVaR, EPD, mean and CV table)

        From great.pres_maker.df_to_tikz

        keyword args:

            scale           picks up self.tikz_scale; scale applied to whole
                            table - default 0.717
            height          row height, rec. 1 (em)
            column_sep      col sep in em
            row_sep         row sep in em
            container_env   table, figure or sidewaysfigure
            color           color for text boxes (helps debugging)
            extra_defs      TeX defintions and commands put at top of table,
                            e.g., \\centering
            lines           lines below these rows, -1 for next to last row
                            etc.; list of ints
            post_process    e.g., non-line commands put at bottom of table

            latex           arguments after \begin{table}[latex]
            caption         text for caption

        Previous version see great.pres_maker
        Original version see: C:\\S\\TELOS\\CAS\\AR_Min_Bias\\cvs_to_md.py

        :param column_sep:
        :param row_sep:
        :param figure:
        :param extra_defs:
        :param post_process:
        :param label:
        :return:
        """
        # local variable - with all formatters already applied
        df = self.apply_formatters(self.raw_df.copy(), mode='raw')
        caption = self.caption
        label = self.label
        # prepare label and caption
        if label == '':
            lt = ''
            label = ''
        else:
            lt = label
            label = f'\\label{{{label}}}'
        if caption == '':
            if lt != '':
                logger.info(
                    f'You have a label but no caption; the label {label} will be ignored.')
            caption = '% caption placeholder'
        else:
            caption = f'\\caption{{{self.caption}}}\n{label}'

        if not df.columns.is_unique:
            # possible index/body column interaction
            raise ValueError('tikz routine requires unique column names')
# {extra_defs}
        # centering handled by quarto
        header = """
\\begin{{{container_env}}}{latex}
{caption}
\\centering{{
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
}}   % close centering
\\end{{{container_env}}}
"""

        # always a good idea to do this...need to deal with underscores, %
        # and it handles index types that are not strings
        df = GT.clean_index(df)
        if not np.all([i == 'object' for i in df.dtypes]) and not df.empty:
            logger.warning('cols of df not all objects (expect all obs at this '
                           'point): ', df.dtypes, sep='\n')
        # make sure percents are escaped, but not if already escaped
        df = df.replace(r"(?<!\\)%", r"\%", regex=True)

        # pres_maker.df_to_tikz code line 1931
        if self.show_index:
            nc_index = df.index.nlevels
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", category=pd.errors.PerformanceWarning)
                df = df.reset_index(
                    drop=False, col_level=df.columns.nlevels - 1)
            if sparsify:
                if hrule is None:
                    hrule = set()
            for i in range(sparsify):
                # TODO update to new sparsify!!
                df.iloc[:, i], rules = GT.sparsify_old(df.iloc[:, i])
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
        logger.info(
            f'rows in columns {nr_columns}, columns in index {nc_index}')

        # internal TeX code (same as HTML code)
        matrix_name = self.df_id

        # note this happens AFTER you have reset the index...need to pass
        # number of index columns
        # have also converted everything to formatted strings
        # estimate... originally called guess_column_widths, with more parameters
        colw, tabs = GT.estimate_column_widths(df, self.max_table_width, nc_index=nc_index, scale=self.tikz_scale, equal=self.equal)  # noqa
        if self.debug:
            print(f'Make TikZ Input {self.tabs=}\nComputed {tabs=}')
        if self.tabs is not None:
            if len(tabs) == len(self.tabs):
                tabs = self.tabs
            elif len(self.tabs) == 1:
                tabs = self.tabs * len(tabs)
            else:
                logger.error(
                    f'{self.tabs=} must be None, a single number, or a list of numbers of the correct length. Ignoring.')
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
        sio.write(header.format(container_env=container_env,
                                caption=caption,
                                extra_defs=extra_defs,
                                scale=self.tikz_scale,
                                column_sep=column_sep,
                                row_sep=row_sep,
                                latex=latex,
                                debug=debug))

        # table header
        # title rows, start with the empty spacer row
        i = 1
        sio.write(
            f'\trow {i}/.style={{nodes={{text=black, anchor=north, inner ysep=0, text height=0, text depth=0}}}},\n')
        for i in range(2, nr_columns + 2):
            sio.write(
                f'\trow {i}/.style={{nodes={{text=black, anchor=south, inner ysep=.2em, minimum height=1.3em, font=\\bfseries}}}},\n')

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
        sio.write(
            f'\tcolumn {i+1:>2d}/.style={{text height=0.9em, text depth=0.2em, nosep, text width=0em}}')
        sio.write('\t}]\n')

        sio.write("\\matrix ({matrix_name}) [table, ampersand replacement=\\&]{{\n".format(
            matrix_name=matrix_name))

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
        def python_2_tex(x): return x + nr_columns + \
            2 if x >= 0 else nr + x + 3
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
        sio.write(
            f'\\path[draw, {ls}] ({matrix_name}-{ln}-1.south west)  -- ({matrix_name}-{ln}-{nc+1}.south east);\n')

        for ln in hrule:
            ls = 'thick' if ln == nr + nr_columns + \
                1 else ('semithick' if ln == 1 + nr_columns else 'very thin')
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
            logger.debug(
                f'Generated vlines {mi_vrules}; already written {written}')
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

        sio.write(footer.format(container_env=container_env,
                  post_process=post_process))

        return sio.getvalue()

    @staticmethod
    def estimate_column_widths(df, target_width, nc_index, scale, equal=False):
        """
        Estimate sensible column widths for the dataframe [in what units?]

        Internal variables:
            mxmn   affects alignment: are all columns the same width?

        TODO: de-TeX-ification will mess up how the tex table is printed...
            but one rarely looks at that.

        :param df:
        :param nc_index: number of columns in the index...these are not counted as "data columns"
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
        if df.empty:
            return colw, tabs
        nl = nc_index
        for i, c in enumerate(df.columns):
            # figure width of the column labels; if index c= str, if MI then c = tuple
            # cw is the width of the column header/title
            if type(c) == str:
                if i < nl:
                    cw = GT.text_display_len(c)
                else:
                    # for data columns look at words rather than whole phrase
                    cw = max(map(GT.text_display_len, c.split(' ')))
                    # logger.info(f'leng col = {len(c)}, longest word = {cw}')
            else:
                # column name could be float etc. or if multi index a tuple
                try:
                    if isinstance(c, tuple):
                        # multiindex: join and split into words and take length of each word
                        words = ' '.join(c).split(' ')
                        cw = max(
                            map(lambda x: GT.text_display_len(str(x)), words))
                    else:
                        cw = max(map(lambda x: GT.text_display_len(str(x)), c))
                    # print(f'{c}: {cw=} no error')
                except TypeError:
                    # not a MI, float or something
                    cw = GT.text_display_len(str(c))
                    # print(f'{c}: {cw=} WITH error')
            headw[c] = cw
            # now figure the width of the elements in the column
            # mxmn is used to determine whether to center the column (if all the same size)
            if df.dtypes.iloc[i] == object:
                # weirdness here were some objects actually contain floats, str evaluates to NaN
                # and picks up width zero
                try:
                    lens = df.iloc[:, i].map(
                        lambda x: GT.text_display_len(str(x)))
                    colw[c] = lens.max()
                    mxmn[c] = (lens.max(), lens.min())
                except Exception as e:
                    logger.error(
                        f'{c} error {e} DO SOMETHING ABOUT THIS...if it never occurs dont need the if')
                    colw[c] = df[c].str.len().max()
                    mxmn[c] = (df[c].str.len().max(), df[c].str.len().min())
            else:
                lens = df.iloc[:, i].map(lambda x: GT.text_display_len(str(x)))
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
        target_width = target_width * scale - index_width
        if data_width and data_width / target_width < 0.9:
            # don't rescale above 1:1 - don't want too large
            rescale = min(1 / scale, target_width / data_width)
            tabs = [w if i < nl else w * rescale for i, w in enumerate(tabs)]
            logger.info(f'Rescale {rescale} applied; tabs = {tabs}')
            # print(f'Rescale {rescale} applied; tabs = {tabs}')
        # print(f'{colw.values()=}\n{tabs=}')
        return colw, tabs

    @staticmethod
    def sparsify(df, cs):
        out = df.copy()
        for i, c in enumerate(cs):
            mask = df[cs[:i + 1]].ne(df[cs[:i + 1]].shift()).any(axis=1)
            out.loc[~mask, c] = ''
        return out

    @staticmethod
    def sparsify_old(col):
        """
        sparsify col values, col a pd.Series or dict, with items and accessor
        column results from a reset_index so has index 0,1,2... this is relied upon.
        TODO: this doesn't work if there is a change in a higher level but not this level
        """
        # fix error with empty dfs
        new_col = col.copy()
        rules = []
        if col.empty:
            return new_col, rules
        last = col[0]
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

        :param n: input name, str
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
        r"""
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
        ef = pd.io.formats.format.EngFormatter(neng, True)  # noqa
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
        p.write_text(soup.prettify(), encoding='utf-8')
        logger.info(f'Saved to {p}')

    @staticmethod
    def parse_markdown_table_and_caption(txt: str) -> tuple[str, str | None]:
        """
        Parses a Markdown table and an optional caption from a given string,
        handling cases where only the caption is present.

        Args:
            txt: The input string.

        Returns:
            A tuple containing the table string (empty if not found) and the caption string (or None if no caption).
        """
        table_match = re.search(r"((?:\|.*\|\s*(?:\n|$))+)", txt, re.DOTALL)
        caption_match = re.search(
            r"^(?:table)?:\s*(.+)", txt, re.MULTILINE + re.IGNORECASE)

        table_part = table_match.group(1).strip() if table_match else ""
        caption_part = caption_match.group(1) if caption_match else ""

        return table_part.strip(), caption_part.strip()

    @staticmethod
    def md_to_df(txt):
        """Convert markdown text string table to DataFrame."""
        # extract table and optional caption part
        table, caption = GT.parse_markdown_table_and_caption(txt)
        m = re.search(r'\{#(tbl[:a-zA-Z0-9_-]+)\}', caption)
        if m:
            label = m.group(1)
            if label != '':
                # remove from caption
                caption = caption.replace(f'{{#{label}}}', '').strip()
        else:
            label = ''
        # print(f'{caption = } and {label = }')
        if table == '':
            raise ValueError('Bad markdown table')

        # remove starting and ending | in each line (optional anyway)
        txt = re.sub(r'^\||\|$', '', table, flags=re.MULTILINE)
        txt = txt.split('\n')
        # remove starting and ending *'s added by hand - but try to avoid * within headings!
        txt[0] = '|'.join([re.sub(r'^\*\*?|\*\*?$', '', i.strip())
                          for i in txt[0].split('|')])

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
        txt = [[j.strip() for j in i.split('|')] for i in txt]
        df = pd.DataFrame(txt).T
        df = df.set_index(0)
        df = df.T
        return df, aligners, caption, label

    @staticmethod
    def to_text_table(
        df: pd.DataFrame,
        data_col_widths: list[int],
        data_col_aligns: list[str],
        *,
        index_levels: int = 1,
        fmt: TableFormat = GT_Format
    ) -> str:
        """
        Render self.df as a wrapped, boxed table.

        Output like tabulate's mixed_grid with support for:
        - Multi-level column headers (always shown, bottom-aligned, can wrap)
        - Split index vs. body section with heavy vertical separator
        - Per-column width and alignment
        - Wrapped body cells with top alignment

        Custom code to print a dataframe to text.

        pd.DataFrame.to_string uses tabulate.tabulate which is hard to
        control. This modoule provides similar functionality with greater
        control over column widths and the ability to demark the index
        columns.

        Tabulate table formats are defined in namedtuples found in tf:

        from tabulate import _table_formats as tf

        This is a dict and e.g.,

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
        'latex_raw', 'latex_booktabs', 'latex_longtable', 'tsv', 'textile',
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

        Returns:
            str: A fully formatted table as a string (useful for print, logs, or files).
        """
        buf = StringIO()

        def _write_line(line: str) -> None:
            """Writes a line to the buffer followed by a newline."""
            buf.write(line + '\n')

        def _format_cell(text: str, width: int, align: str) -> list[str]:
            """
            Formats a single cell, wrapping text and applying padding and alignment.
            Returns a list of strings, each representing a line of the cell.
            """
            lines = wrap(str(text), width=width) or ['']
            padded_width = width + 2 * fmt.padding
            return [
                (" " * fmt.padding)
                + (line.ljust(width) if align == 'left'
                   else line.center(width) if align == 'center'
                   else line.rjust(width)) +
                (" " * fmt.padding)
                for line in lines
            ]

        def _make_horizontal_line(line_fmt: Line, col_widths: list[int]) -> str:
            """Constructs a full horizontal line for the table."""
            parts = []
            for i, w in enumerate(col_widths):
                total = w + 2 * fmt.padding
                if index_levels and i == index_levels:
                    parts.append(line_fmt.index_sep)
                elif i > 0:
                    parts.append(line_fmt.sep)
                parts.append(line_fmt.hline * total)
            return f"{line_fmt.begin}{''.join(parts)}{line_fmt.end}"

        def _make_data_row(row_fmt: DataRow, line_cells: list[str]) -> str:
            """Constructs a single data row from formatted cell strings."""
            parts = []
            for i, cell in enumerate(line_cells):
                if index_levels and i == index_levels:
                    parts.append(row_fmt.index_sep)
                elif i > 0:
                    parts.append(row_fmt.sep)
                parts.append(cell)
            return f"{row_fmt.begin}{''.join(parts)}{row_fmt.end}"

        def _render_header_level(wrapped_cells: list[list[str]], level_widths: list[int]) -> list[str]:
            """
            Renders a single level of the header, ensuring cells are bottom-aligned.
            Returns a list of strings, each representing a line of the header.
            """
            max_height = max(len(c) for c in wrapped_cells)
            padded_cells = [
                [' ' * (w + 2 * fmt.padding)] * (max_height - len(cell)) + cell
                for cell, w in zip(wrapped_cells, level_widths)
            ]
            return [_make_data_row(fmt.headerrow, [col[i] for col in padded_cells]) for i in range(max_height)]

        col_levels = df.columns.nlevels
        col_tuples = df.columns if col_levels > 1 else [
            (c,) for c in df.columns]

        # Step 1: format each level of the column headers (one header line per level)
        # header alignment is left in index and center in body
        index_col_aligns = [
            'left' if i < index_levels else 'center' for i in range(len(data_col_aligns))]
        _write_line(_make_horizontal_line(fmt.lineabove, data_col_widths))
        # collect all wrapped + bottom-aligned rows for each level
        for level in range(col_levels):
            level_texts = [str(t[level] if level < len(t) else '')
                           for t in col_tuples]
            wrapped_cells = [_format_cell(txt, w, a) for txt, w, a in zip(
                level_texts, data_col_widths, index_col_aligns)]
            level_rows = _render_header_level(wrapped_cells, data_col_widths)
            for row in level_rows:
                _write_line(row)
            if level < col_levels - 1:
                _write_line(_make_horizontal_line(
                    fmt.linebetweenrows, data_col_widths))
        _write_line(_make_horizontal_line(
            fmt.linebelowheader, data_col_widths))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            data_cells = [
                _format_cell(val, w, a)
                for val, w, a in zip(row.values, data_col_widths, data_col_aligns)
            ]
            max_height = max(len(c) for c in data_cells)
            padded = [
                c + [' ' * (w + 2 * fmt.padding)] * (max_height - len(c))
                for c, w in zip(data_cells, data_col_widths)
            ]
            for i in range(max_height):
                _write_line(_make_data_row(
                    fmt.datarow, [col[i] for col in padded]))

            if row_idx < len(df) - 1:
                _write_line(_make_horizontal_line(
                    fmt.linebetweenrows, data_col_widths))
            else:
                _write_line(_make_horizontal_line(
                    fmt.linebelow, data_col_widths))

        return buf.getvalue()

    @staticmethod
    def _estimate_math_width(tex: str) -> int:
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]|.', tex)
        width = 0
        for tok in tokens:
            if tok.startswith('\\'):
                name = tok[1:]
                if name in GT.TEX_SIMPLE_GLYPHS:
                    width += 1
                elif name in GT.TEX_WIDE:
                    width += 3
                elif name in GT.TEX_SPACING:
                    width += 1
                else:
                    width += 2  # unknown control sequences
            elif tok in '{}^_':
                continue  # grouping, sub/superscripts: ignore
            else:
                width += 1
        return width

    @staticmethod
    def text_display_len(s: str) -> int:
        """Estimate text display length of a string allowing for TeX constructs."""
        # note you DO WANT SPACES! So, no strip applied ever.
        if s.find('$') < 0:
            return len(s)
        parts = re.split(r'(\$\$.*?\$\$)|(\$.*?\$)', s)
        total = 0
        for part in parts:
            if part is None:
                continue
            if part.startswith('$$') and part.endswith('$$'):
                total += GT._estimate_math_width(part[2:-2])
            elif part.startswith('$') and part.endswith('$'):
                total += GT._estimate_math_width(part[1:-1])
            else:
                total += len(part)
        return total

    @staticmethod
    def make_rich_table(
        df,
        column_widths,
        column_alignments=None,
        num_index_columns=0,
        title=None,
        show_lines=False,
        box_style=box.SIMPLE_HEAVY,
    ):
        """
        Render a preformatted DataFrame as a Rich table.

        Parameters:
            df (pd.DataFrame): DataFrame with all string values.
            column_widths (dict or list): Widths by column name or position.
            column_alignments (dict or list): Alignments ('left', 'center', 'right').
            num_index_columns (int): Number of left-most columns to treat as index-like.
            title (str): Optional title.
            show_lines (bool): Add row separator lines.
            box_style (rich.box.Box): Border style (see below).
        """
        colnames = list(df.columns)

        if isinstance(column_widths, list):
            column_widths = {colnames[i]: w for i,
                             w in enumerate(column_widths)}

        if column_alignments is None:
            column_alignments = {}
        elif isinstance(column_alignments, list):
            column_alignments = {
                colnames[i]: a for i, a in enumerate(column_alignments)}

        table = Table(title=title,
                      box=box_style,
                      show_lines=show_lines,
                      expand=True)

        for i, col in enumerate(colnames):
            is_index = i < num_index_columns
            table.add_column(
                header=str(col),
                width=column_widths.get(col, None),
                justify=column_alignments.get(col, "left"),
                style="dim" if is_index else None,
                header_style="bold dim" if is_index else "bold",
                no_wrap=False,
                overflow="fold",
                vertical="middle",
                # divider=divider,
            )

        for _, row in df.iterrows():
            table.add_row(*row.tolist())

        return table

    def rich_table(self, console, box_style=box.SQUARE):
        """Render to a rich table using Console object console."""
        # hijack max table width
        mtw = self.max_table_width
        tw_mode = self.table_width_mode
        self.table_width_mode = 'explicit'
        self.max_table_width = console.width
        # figure col widths and aligners
        cw_df = self.make_column_width_df()
        cw = cw_df['recommended']
        aligners = cw_df['alignment']
        show_lines = self.hrule_widths[0] > 0

        self._rich_table = table = GT.make_rich_table(self.df,
                                                      cw,
                                                      aligners,
                                                      num_index_columns=self.nindex,
                                                      title=self.caption,
                                                      show_lines=show_lines,
                                                      box_style=box_style)

        self.max_table_width = mtw
        self.table_width_mode = tw_mode
        return table
