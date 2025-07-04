"""
Core rendering logic for GreaterTables.

Defines the `GreaterTables` class, which formats and renders pandas DataFrames
to HTML, plain text, or LaTeX output using a validated configuration model.

This is the main entry point for rendering logic. See `gtconfig.py` for configuration schema.
"""

from decimal import InvalidOperation
from io import StringIO
from itertools import groupby
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Optional, Union, Literal
import warnings
import yaml

from bs4 import BeautifulSoup
from cachetools import LRUCache
import numpy as np
import pandas as pd
from pandas.errors import IntCastingNaNError
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, \
    is_float_dtype   # , is_numeric_dtype
from pydantic import ValidationError
from rich import box
from IPython.display import display, SVG

from . enums import Breakability
from . config import Configurator
from . hasher import df_short_hash
from . etcher import Etcher
from . utilities import *

# turn off this fuck-fest
pd.set_option('future.no_silent_downcasting', True)
# pandas complaining about casting columns eg putting object in float column
warnings.simplefilter(action='ignore', category=FutureWarning)


# GPT recommended approach
logger = logging.getLogger(__name__)

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
      ``config.header_row``
    * A string is  assumed to be a pipe-separated markdown table which is
      converted to a ``DataFrame`` setting aligners per the alignment row
    * All other input types are an error

    The input ``df`` must have unique column names. It is then copied into
    ``self.df`` which will be changed and ``self.raw_df`` for reference.
    The copy is hashed for the table name.

    **Mangling**

    * If show_index, the index is reset and kept, so that all columns are on an
      config.equal footing
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
    :param config.default_integer_str: format f-string for integers, default
      value '{x:,d}'
    :param config.default_float_str: format f-string for floats, default
      value '{x:,.3f}'
    :param config.default_date_str: format f-string for dates, default '%Y-%m-%d'.
      NOTE: no braces or x!
    :param config.default_ratio_str: format f-string for ratios, default '{x:.1%}'
    :param config.table_float_format: None or format string for floats in the
      table format function, applied to entire table, default None
    :param config.table_hrule_width: width of the table top, botton and header
      hrule, default 1
    :param config.table_vrule_width: width of the table vrule, separating the
      index from the body, default 1
    :param config.hrule_widths: None or tuple of three ints for hrule widths
      (for use with multiindexes)
    :param config.vrule_widths: None or tuple of three ints for vrule widths
      (for use when columns have multiindexes)
    :param config.sparsify: if True, config.sparsify the index columns, you almost always
      want this to be true!
    :param config.sparsify_columns: if True, config.sparsify the columns, default True,
      generally a better look, headings centered in colspans
    :param config.spacing: 'tight', 'medium', 'wide' to quickly set cell padding.
      Medium is default (2, 10, 2, 10).
    :param config.padding_trbl: None or tuple of four ints for padding, in order
      top, right, bottom, left.
    :param config.tikz_scale: scale factor applied to tikz LaTeX tables.
    :param config.font_body: font size for body text, default 0.9. Units in em.
    :param config.font_head: font size for header text, default 1.0. Units in em.
    :param config.font_caption: font size for caption text, default 1.1.
      Units in em.
    :param config.font_bold_index: if True, make the index columns bold,
      default False.
    :param config.pef_precision: precision (digits after period) for pandas
      engineering format, default 3.
    :param config.pef_lower: apply engineering format to floats with absolute
      value < 10**config.pef_lower; default -3.
    :param config.pef_upper: apply engineering format to floats with absolute
      value > 10**config.pef_upper; default 6.
    :param config.cast_to_floats: if True, try to cast all non-integer, non-date
      columns to floats
    :param config.header_row: True: use first row as headers; False no headings.
      Default True
    :param config.tabs: None or list of column widths in characters or a common
      int or float width. (It is converted into em; one character is about
      0.5em on average; digits are exactly 0.5em.) If None, will be calculated.
      Default None.
    :param config.equal: if True, set all column widths config.equal. Default False. Maybe
      ignored, depending on computed ideal column widths.
    :param config.caption_align: for the caption
    :param config.large_ok: signal that you are intentionally applying to a large
      dataframe. Sub-classes may restrict or apply .head() to df.
    :param config.max_str_length: maximum displayed length of object types, that
      are cast to strings. Eg if you have nested DataFrames!
    :param str_table_fmt: table border format used for string output
      (markdown), default mixed_grid DEPRECATED??
    :param config.table_width_mode:
        'explicit': set using config.max_table_width_em
        'natural': each cell on one line (can be very wide with long strings)
        'breakable': wrap breakable cells (text strings) at word boundaries
          to fit longest word
        'minimum': wrap breakable and ok-to-break (dates) cells
    :param config.table_width_header_adjust: additional proportion of table width
      used to balance header columns.
    :param config.table_width_header_relax: extra spaces allowed per column heading
      to facilitate better column header wrapping.
    :param config.max_table_width_em: max table width used for markdown string output,
      default 200; width is never less than minimum width. Padding (3 chars
      per row plus 1) consumed out of config.max_table_width_em in string output mode.
    :param config.debug: if True, add id to caption and use colored lines in table,
      default False.
    """

    def __init__(
        self,
        df,
        *,
        caption='',
        label='',
        aligners: dict[str, callable] | None = None,
        formatters: dict[str, callable] | None = None,
        tabs: Optional[Union[list[float], float, int]] | None = None,
        unbreakable=None,
        ratio_cols=None,
        year_cols=None,
        date_cols=None,
        raw_cols=None,
        show_index=True,
        #
        config: Configurator | None = None,
        config_path: Path | None = None,
        **overrides,
    ):
        if config and config_path:
            raise ValueError(
                "Pass either 'config' or 'config_path', not both.")

        if config:
            base_config = config
        elif config_path:
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                base_config = Configurator.model_validate(raw)
            except (ValidationError, OSError) as e:
                raise ValueError(
                    f"Failed to load config from {config_path}") from e
        else:
            base_config = Configurator()

        # access through config
        # update and validate; need to merge to avoid repeated args
        # merged = dict(base_config.model_dump(), **overrides)
        merged = base_config.model_dump() | overrides
        self.config = Configurator(**merged)
        # no validation
        # self.config = base_config.model_copy(update=overrides)

        # deal with alternative input modes for df: None, DataFrame, Series, markdown text table
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
            if config.header_row:
                # Set first row as column names
                df.columns = df.iloc[0]
                # Drop first row and reset index
                df = df[1:].reset_index(drop=True)
        elif isinstance(df, str):
            df = df.strip()
            if df == '':
                df = pd.DataFrame([])
            else:
                df, aligners, caption, label = MD2DF.md_to_df(df)
                show_index = False
        else:
            raise ValueError(
                'df must be a DataFrame, a list of lists, or a markdown table string')

        if len(df) > self.config.large_warning and not self.config.large_ok:
            raise ValueError(
                'Large dataframe (>50 rows) and config.large_ok not set to true...do you know what you are doing?')

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

        if self.caption != '' and self.config.debug:
            self.caption += f' (id: {self.df_id})'
        # self.max_str_length = max_str_length
        # before messing
        self.show_index = show_index
        self.nindex = self.df.index.nlevels if self.show_index else 0
        self.ncolumns = self.df.columns.nlevels
        self.ncols = self.df.shape[1]
        self.dt = self.df.dtypes

        # reset index to put all columns on an config.equal footing, but note number ofindex cols
        with warnings.catch_warnings():
            if self.show_index:
                warnings.simplefilter(
                    "ignore", category=pd.errors.PerformanceWarning)
                self.df = self.df.reset_index(
                    drop=False, col_level=self.df.columns.nlevels - 1)
            # want the new index to be ints - that is not default if old was multiindex
            self.df.index = np.arange(self.df.shape[0], dtype=int)
        self.index_change_level = Indexing.changed_column(
            self.df.iloc[:, :self.nindex])
        if self.ncolumns > 1:
            # will be empty rows above the index headers
            self.index_change_level = pd.Series(
                [i[-1] for i in self.index_change_level])

        self.column_change_level = Indexing.changed_level(self.raw_df.columns)

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
        if self.config.default_formatter is None:
            self.default_formatter = self._default_formatter
        else:
            assert callable(
                self.config.default_formatter), 'config.default_formatter must be callable'

            def wrapped_default_formatter(x):
                try:
                    return self.config.default_formatter(x)
                except ValueError:
                    return str(x)
            self.default_formatter = wrapped_default_formatter

        # cast as much as possible to floats
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=pd.errors.PerformanceWarning)
            if self.config.cast_to_floats:
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

        if tabs is None:
            self.tabs = None
        elif isinstance(tabs, (int, float)):
            self.tabs = (tabs,) * (self.nindex + self.ncols)
        elif isinstance(tabs, (np.ndarray, pd.Series, list, tuple)):
            if len(tabs) == self.nindex + self.ncols:
                self.tabs = tabs  # Already iterable and right length, self.tabs = as is
            else:
                logger.error(
                    f'{self.tabs=} has wrong length. Ignoring.')
                self.tabs = None
        else:
            logger.error(
                f'{self.tabs=} must be None, a single number, or a list of '
                'numbers of the correct length. Ignoring.')
            self.tabs = None

        if self.config.padding_trbl is not None:
            padding_trbl = self.config_padding_trbl
        elif self.config.padding_trbl is None:
            if self.config.spacing == 'tight':
                padding_trbl = (0, 5, 0, 5)
            elif self.config.spacing == 'medium':
                padding_trbl = (2, 10, 2, 10)
            elif self.config.spacing == 'wide':
                padding_trbl = (4, 15, 4, 15)
            else:
                raise ValueError(
                    'config.spacing must be tight, medium, or wide or tuple of four ints.')
        # pydantic will see to it this is OK
        self.padt, self.padr, self.padb, self.padl = padding_trbl

        # because of the problem of non-unique indexes use a list and
        # not a dict to pass the formatters to to_html
        self.max_table_width_em = self.config.max_table_inch_width * 72 / self.config.table_font_pt_size
        self._pef = None
        self._df_formatters = None
        self.df_style = ''
        self.df_html = ''
        self._clean_html = ''
        self._clean_tex = ''
        self._rich_table = None
        self._string = ''
        self._df_html_text = ""
        self._df_style_text = ""
        self._cache = LRUCache(20)
        self._text_knowledge_df = None
        self._html_knowledge_df = None
        self._tex_knowledge_df = None
        self._knowledge_dfs = None
        # finally config.sparsify and then apply formaters
        # this radically alters the df, so keep a copy for now...
        self.df_pre_applying_formatters = self.df.copy()
        self.df = self.apply_formatters(self.df)
        # config.sparsify
        if self.config.sparsify and self.nindex > 1:
            self.df = Sparsify.sparsify(self.df, self.df.columns[:self.nindex])
            # for c in self.df.columns[:self.nindex]:
            #     # config.sparsify returns some other stuff...
            #     self.df[c], _ = GT.config.sparsify(self.df[c])
        # make final tex and html versions
        if self.config.tex_to_html is not None:
            # NEED TO WORK ON INDEXES TOO
            self.df_html = self.df.map(self.config.tex_to_html)
        else:
            self.df_html = self.df
        if self.config.tikz_escape_tex:
            self.df_tex = Escaping.escape_df_tex(self.df)
        else:
            self.df_tex = self.df

    def __repr__(self):
        """Basic representation."""
        return f"GT(df_id={self.df_id})"

    def __str__(self):
        """String representation, for print()."""
        return self.make_string()

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

    def cache_get(self, key):
        """Retrieve item from cache."""
        return self._cache.get(key, None)

    def cache_set(self, key, value):
        """Add item to cache."""
        self._cache[key] = value

    def cols_from_regex(self, regex):
        """
        Return columns matching a regex.

        For Index and MultiIndex. Operates on ``self.df`` and includes
        index (if ``show_index``) and columns of input dataframe. Search
        applies to any level of the index. Case sensitive.
        """
        pattern = re.compile(regex)
        matching_cols = [
            col for col in self.df.columns
            if any(pattern.search(str(level))
                for level in (col if isinstance(col, tuple) else (col,)))
        ]
        return matching_cols
        # return [col for col in self.df.columns if isinstance(col, str) and re.search(regex, col)]

    # define the default and easy formatters ===================================================
    def default_ratio_formatter(self, x):
        """Ratio formatter."""
        try:
            return self.config.default_ratio_str.format(x=x)
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
            return dt.strftime(self.config.default_date_str)
        except Exception:
            logger.error("date error with %s", x)
            return str(x)

    def default_integer_formatter(self, x):
        """Integer formatter."""
        try:
            return self.config.default_integer_str.format(x=x)
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

    def _default_formatter(self, x):
        """Default universal formatter for other types."""
        try:
            f = float(x)
        except (TypeError, ValueError):
            s = str(x)
            return s if self.config.max_str_length < 0 else s[:self.config.max_str_length]

        if self.default_float_formatter:
            return self.default_float_formatter(f)

        if np.isinf(f) or np.isnan(f):  # clearer handling of weird float cases
            return str(x)

        if f.is_integer():
            return self.config.default_integer_str.format(x=int(f))
        else:
            return self.config.default_float_str.format(x=f)

    def pef(self, x):
        """Pandas engineering format."""
        if self._pef is None:
            self._pef = pd.io.formats.format.EngFormatter(accuracy=self.config.pef_precision, use_eng_prefix=True)   # noqa
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
        # pl = 10. ** self.config.pef_lower
        # pu = 10. ** self.config.pef_upper
        pl, pu = 10. ** self.config.pef_lower, 10. ** self.config.pef_upper
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
            if self.config.table_float_format is not None:
                if callable(self.config.table_float_format):
                    # wrap in error protections
                    def ff(x):
                        try:
                            return self.config.table_float_format(x=x)
                        except ValueError:
                            return str(x)
                        except Exception as e:
                            logger.error(f'Custom float function raised {e=}')
                    self.default_float_formatter = ff
                else:
                    if type(self.config.table_float_format) != str:
                        raise ValueError(
                            'config.table_float_format must be a string or a function')
                    fmt = self.config.table_float_format

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
            # self._df_formatters is now a list of length config.equal to cols in df
            if len(self._df_formatters) != self.df.shape[1]:
                raise ValueError(
                    f'Something wrong: {len(self._df_formatters)=} != {self.df.shape=}')
        return self._df_formatters

    @staticmethod
    def apply_formatters_work(df, formatters):
        """Apply formatters to a DataFrame."""
        try:
            new_df = pd.DataFrame({i: map(f, df.iloc[:, i])
                                   for i, f in enumerate(formatters)})
        except TypeError:
            print('NASTY TYPE ERROR')
            raise
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

    @property
    def text_knowledge_df(self):
        """Uber source of information for text formatting."""
        if self._text_knowledge_df is None:
            self._text_knowledge_df = self.estimate_column_widths_by_mode('text')
        return self._text_knowledge_df

    @property
    def html_knowledge_df(self):
        """Uber source of information for html formatting."""
        if self._html_knowledge_df is None:
            self._html_knowledge_df = self.estimate_column_widths_by_mode('html')
        return self._html_knowledge_df

    @property
    def tex_knowledge_df(self):
        """Uber source of information for tex formatting."""
        if self._tex_knowledge_df is None:
            # seems this is unlikely to be a good idea!
            # if (all(self.df_tex.index == self.df_html.index)
            #     and all(self.df_tex.columns == self.df_html.columns)
            #     and all(self.df_tex == self.df_html)):
            #     self._tex_knowledge_df = self.html_knowledge_df
            # else:
            self._tex_knowledge_df = self.estimate_column_widths_by_mode('tex')
        return self._tex_knowledge_df

    @property
    def knowledge_dfs(self):
        if self._knowledge_dfs is None:
            self._knowledge_dfs = pd.concat((self.text_knowledge_df.T,
                        self.html_knowledge_df.T, self.tex_knowledge_df.T),
                        keys=['text','html', 'tex'], names=['mode', 'measure'])
            self._knowledge_dfs['Total'] = self._knowledge_dfs.fillna(0.).apply(
                lambda row: sum(x for x in row if pd.api.types.is_number(x)), axis=1)
            idx = self._knowledge_dfs.query('Total == 0').index
            self._knowledge_dfs.loc[idx, 'Total'] = ''
            self._knowledge_dfs = self._knowledge_dfs.fillna('')
        return self._knowledge_dfs

    def width_report(self):
        """Return a report summarizing the width information."""
        natural = self.text_knowledge_df.natural_width.sum()
        minimum = self.text_knowledge_df.minimum_width.sum()
        text = self.text_knowledge_df.recommended.sum()
        h = self.html_knowledge_df.recommended.sum()
        tex =  self.tex_knowledge_df.recommended.sum()
        tikz = self.tex_knowledge_df.tikz_colw.sum()
        mtw = self.max_table_width_em
        mtiw = self.config.max_table_inch_width
        pts = self.config.table_font_pt_size
        bit = pd.DataFrame({
                        'text natural': self.text_knowledge_df.natural_width,
                        'text minimum': self.text_knowledge_df.minimum_width,
                        'text recommended': self.text_knowledge_df.recommended,
                        'html recommended': self.html_knowledge_df.recommended,
                        'tex recommended': self.tex_knowledge_df.recommended,
                        'tikz recommended': self.tex_knowledge_df.tikz_colw,
        }).fillna(0)
        ser = pd.Series({
                        'text natural': natural,
                        'text minimum': minimum,
                        'text recommended': text,
                        'html recommended': h,
                        'tex recommended': tex,
                        'tikz recommended': tikz,
        })
        bit.loc['total', :] = ser
        print(f"requested width = {mtw} em\n"
              f"max tbl inch w  = {mtiw} inches\n"
              f"font pts        = {pts} pts\n"
              f"width in em chk = {mtiw * 72 / pts} em\n"
              f"width mode      = {self.config.table_width_mode}\n"
              f"header relax    = {self.config.table_width_header_adjust}\n"
              f"header chars    = {self.config.table_width_header_relax}")
        return bit

    def estimate_column_widths_by_mode(self, mode):
        r"""
        Return dataframe of width information: three modes for text, html, and tex.

        Mode adjusts which df is used and how widths are estimated

        * text -> self.df and len = str.len
        * html -> self.df_html and len =
        * tex  -> self.df_tex and len =

        Returned dataframe has columns named mode_xxx, where xxx can be

        * natural: max len by col
        * minimum width = max length given breaks
        * acceptable = allowing for break type by column

        * head_natural, head_min, head_acceptable for the heading

        * raw_recommended
        * header_adjustment
        * recommended

        pat and iso_date_split regex explanation:

            # re.split(r'(?<=[\s.,:;!?()\[\]{}\-\\/|])\s*', text)
            # (?<=...) is a lookbehind to preserve the break character with the left-hand fragment.
            # [\s.,:;!?()\[\]{}\-\\/|] matches common punctuation and separators:
            # \s = whitespace
            # . , : ; ! ? = terminal punctuation
            # () [] {} = brackets
            # \- = dash
            # \\/| = slash, backslash, pipe

        """
        assert mode in ('text', 'html', 'tex'), 'Only html, text and tex modes valid.'
        if mode == 'text':
            df = self.df
            len_function = len
        elif mode == 'html':
            df = self.df_html
            len_function = TextLength.text_display_len
        else: #  mode == 'tex':
            df = self.df_tex
            len_function = TextLength.text_display_len

        n_row, n_col = df.shape

        # The width if content didn't wrap (single line)
        # Series=dict colname->max width of cells in column
        natural_width = df.map(lambda x: len_function(x.strip())).max(axis=0).to_dict()

        # in text mode: figure out where you can break; pat breaks after punctuation or at -
        pat = r'(?<=[.,;:!?)\]}\u2014\u2013])\s+|--*\s+|\s+'
        iso_date_split = r'(?<=\b\d{4})-(?=\d{2}-\d{2})'
        pat = f'{pat}|{iso_date_split}'

        # Calculate ideal (no wrap) and minimum possible widths for all columns
        # The absolute minimum width each column can take (e.g., longest word for text)
        minimum_width = {}
        header_natural = {}
        header_minimum = {}
        for col_name in df.columns:
            minimum_width[col_name] = (
                df[col_name].str
                .split(pat=pat, regex=True, expand=True)
                .fillna('')
                .map(len_function)
                .max(axis=1)
                .max()
            )
            # ensure is a tuple
            ctuple = col_name if isinstance(col_name, tuple) else (col_name, )
            header_natural[col_name] = max(map(len_function, ctuple))
            header_minimum[col_name] = min(len_function(part) for i in ctuple for part in re.split(pat, str(i)))

        # begin to assemble the parts
        # ans will be the col_width_df; break_penalties needed by all methods
        ans = pd.DataFrame({
            'alignment': [i[4:] for i in self.df_aligners],
            'break_penalties': self.break_penalties,
            'breakability': [x.name for x in self.break_penalties],
            'natural_width': natural_width.values(),
            'minimum_width': minimum_width.values(),
            }, index=df.columns)
        ans['acceptable_width'] = np.where(
            ans.break_penalties == Breakability.ACCEPTABLE, ans.minimum_width, ans.natural_width)
        ans['header_natural'] = header_natural
        ans['header_minimum'] = header_minimum

        if mode in ('html', 'tex'):
            # put in some padding TODO KLUDGE
            ans['natural_width'] += 1
            ans['minimum_width'] += 1
            ans['header_natural'] += 1
            ans['header_minimum'] += 1

        # adjustments and recommendations - these are keyed to text output with padding
        natural, acceptable, minimum = ans.iloc[:, 3:6].sum()
        head_natural, head_minimum = ans.iloc[:, 6:8].sum()

        if mode == 'text':
            # +1 for the pipe | symbol
            PADDING = 2  # per column TODO enhance
            pad_adjustment = (PADDING + 1) * n_col - 1
        else:
            PADDING = 1  # per column TODO enhance
            pad_adjustment =  PADDING * n_col
        if self.config.table_width_mode == 'explicit':
            # target width INCLUDES padding and column marks |
            target_width = self.max_table_width_em - pad_adjustment
        elif self.config.table_width_mode == 'natural':
            target_width = natural + pad_adjustment
        elif self.config.table_width_mode == 'breakable':
            target_width = acceptable + pad_adjustment
        elif self.config.table_width_mode == 'minimum':
            target_width = minimum + pad_adjustment
        logger.info('table_width_mode = %s', self.config.table_width_mode)
        logger.info('config self.max_table_width_em %s', self.max_table_width_em)
        logger.info('target width after column spacer adjustment %s', target_width)

        # extra space for the headers to relax, if useful
        if self.config.table_width_header_adjust > 0:
            max_extra = int(self.config.table_width_header_adjust * target_width)
        else:
            max_extra = 0
        if target_width > natural:
            # everything gets its natural width
            ans['recommended'] = ans['natural_width']
            space = target_width - natural
            logger.info('Space for NATURAL! Spare space = %s', space)
        elif target_width > acceptable:
            # strings wrap
            ans['recommended'] = ans['acceptable_width']
            # use up extra on the ACCEPTABLE cols
            space = target_width - acceptable
            logger.info(
                'Using "breaks acceptable" (dates not wrapped), spare space = %s', space)
        elif target_width > minimum:
            # strings and dates wrap
            ans['recommended'] = ans['minimum_width']
            # use up extra on dates first, then strings
            space = target_width - minimum
            logger.info(
                'Using "minimum" (all breakable incl dates), spare space = %s', space)
        else:
            # OK severely too small
            ans['recommended'] = ans['minimum_width']
            space = target_width - minimum
            logger.warning(
                'Desired width too small for pleasant formatting, table will be too wide by spare space %s < 0.',
                space)
        logger.info(f'{mode=} {target_width=}, {natural=}, {acceptable=}, {minimum=}, {max_extra=}, {space=}')

        # this section tweaks the widths for column headers -> text output only.
        # trust tex and html output to naturally make better decisions about line breaks in the heading.
        if mode == "text" and space > 0 and df.columns.nlevels == 1:
            # text mode only: see if some header tweaks are in order (Index only for now, TODO)
            # Step 1: baseline comes in from code above
            ans['raw_recommended'] = ans['recommended']

            # Step 2: optimize to get rid of intra-line breaks
            if max_extra > 0:
                adj = Width.header_adjustment(df, ans['recommended'], space, max_extra)
                # create new col and populate per GPT
                ans['header_tweak'] = pd.Series(adj)
            else:
                ans['header_tweak'] = 0
            ans['recommended'] = ans['recommended'] + ans['header_tweak']
            # in this case zero out impact of header_natural and header_minimum cos don't want to use them below
            ans['header_natural'] = ans['recommended']
            ans['header_minimum'] = ans['recommended']

        # Step 3 (all modes): distribute remaining shortfall proportionally
        # account for
        # obvs remaining == space if mode is not text
        remaining = target_width - ans['recommended'].sum()
        ans['pre_shortfall_recommended'] = ans['recommended']
        if remaining > 0:
            shortfall = ans[['natural_width', 'header_natural']].max(axis=1) - ans['recommended']
            total_shortfall = shortfall.clip(lower=0).sum()
            if total_shortfall > 0:
                logger.info('total shortfall to allocate after header adjustments = %s', total_shortfall)
                fractions = shortfall.clip(lower=0) / total_shortfall
                ans['proto_recommended'] = ans['recommended'] + np.floor(fractions * remaining).astype(int)
                ans['recommended'] = np.minimum(ans[['natural_width', 'header_natural']].max(axis=1),
                                                ans['proto_recommended'])
            else:
                logger.info('no shortfall to allocate after header adjustments')

        if mode == 'tex':
            # tex mode only need tikz raw size for tex code layout
            tikz_colw = dict.fromkeys(df.columns, 0)
            tikz_headw = dict.fromkeys(df.columns, 0)
            for i, c in enumerate(df.columns):
                # figure width of the column labels
                c0 = c # before we mess around with it, for setting dict values
                if not isinstance(c, tuple):
                    # make it one: now index and multi index on same footing
                    c = (c,)
                # convert to strings
                c = [str(i) for i in c]
                tikz_headw[c0] = max(map(len, c))

                # now figure the width of the elements in the column
                tikz_colw[c0] = df.iloc[:, i].map(lambda x: len(str(x))).max()
            # needed tikz width is greater of two
            for c in df.columns:
                tikz_colw[c] = max(tikz_colw[c], tikz_headw[c])
            # distribute any overage using the measures already done
            ans['tikz_colw'] = tikz_colw
            ans['tikz_colw'] += 2  # for \I

        # in all cases...assemble the answer  with relevant information
        return_columns = [
            'alignment',
            'break_penalties',
            'breakability',
            'natural_width',
            'acceptable_width',
            'minimum_width',
            'header_natural',
            'header_minimum',
            'raw_recommended',
            'header_tweak',
            'pre_space_share_recommended',
            'proto_recommended',
            'recommended',
            'tikz_colw',
            ]
        ans = ans[[i for i in return_columns if i in ans.columns]]
        # need recommended to be > 0
        ans['recommended'] = np.maximum(ans['recommended'], 1)
        return ans

    def make_style(self, tabs):
        """Write out custom CSS for the table."""
        if self.config.debug:
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
        table_hrule = self.config.table_hrule_width
        table_vrule = self.config.table_vrule_width
        # for local use
        padt, padr, padb, padl = self.padt, self.padr, self.padb, self.padl

        style = [f'''
<style>
    #{self.df_id}  {{
    border-collapse: collapse;
    font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
    font-size: {self.config.font_body}em;
    width: auto;
    /* tb and lr
    width: fit-content; */
    margin: 10px auto;
    border: none;
    overflow: auto;
    margin-left: auto;
    margin-right: auto;
    }}
    /* center tables in quarto context
    .greater-table {{
        display: block;
        text-align: center;
    }}
    .greater-table > table {{
        display: inline-table;
    }} */
    /* try to turn off Jupyter and other formats for greater-table
    all: unset => reset all inherited styles
    display: revert -> put back to defaults
    #greater-table * {{
        all: unset;
        display: revert;
    }}
    */
    /* tag formats */
    #{self.df_id} caption {{
        padding: {2 * padt}px {padr}px {padb}px {padl}px;
        font-size: {self.config.font_caption}em;
        text-align: {self.config.caption_align};
        font-weight: normal;
        caption-side: top;
    }}
    #{self.df_id} thead {{
        /* top and bottom of header */
        border-top: {table_hrule}px solid {head_tb};
        border-bottom: {table_hrule}px solid {head_tb};
        font-size: {self.config.font_head}em;
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
        border-top: {self.config.hrule_widths[0]}px solid {h0};
    }}
    #{self.df_id} .grt-hrule-1 {{
        border-top: {self.config.hrule_widths[1]}px solid {h1};
    }}
    #{self.df_id} .grt-hrule-2 {{
        border-top: {self.config.hrule_widths[2]}px solid {h2};
    }}
    /* for the header, there if you have v lines you want h lines
       hence use config.vrule_widths */
    #{self.df_id} .grt-bhrule-0 {{
        border-bottom: {self.config.vrule_widths[0]}px solid {bh0};
    }}
    #{self.df_id} .grt-bhrule-1 {{
        border-bottom: {self.config.vrule_widths[1]}px solid {bh1};
    }}
    #{self.df_id} .grt-vrule-index {{
        border-left: {table_vrule}px solid {v0};
    }}
    #{self.df_id} .grt-vrule-0 {{
        border-left: {self.config.vrule_widths[0]}px solid {v0};
    }}
    #{self.df_id} .grt-vrule-1 {{
        border-left: {self.config.vrule_widths[1]}px solid {v1};
    }}
    #{self.df_id} .grt-vrule-2 {{
        border-left: {self.config.vrule_widths[2]}px solid {v2};
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
        font-size: {self.config.font_head}em;
    }}
    #{self.df_id} .grt-bold {{
        font-weight: bold;
    }}
''']
        # for i, w in enumerate(tabs):
        #     style.append(f'    #{self.df_id} .grt-c-{i} {{ width: {w}em; }}')
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
        bit = self.df_html.T.reset_index(drop=False, allow_duplicates=True)
        idx_header = bit.iloc[:self.nindex, :self.ncolumns]
        columns = bit.iloc[self.nindex:, :self.ncolumns]

        # figure appropriate widths
        tabs = self.html_knowledge_df['recommended'].map(lambda x: np.round(x, 3))

        # set column widths; tabs returns lengths of strings in each column
        tabs = np.array(tabs) + (self.padl + self.padr) / 12
        # this gets stripped out by quarto, so make part of style
        html.append('<colgroup>')
        for w in tabs:
            html.append(f'<col style="width: {w}em;">')
        html.append('</colgroup>')

        # TODO Add header aligners
        # this is TRANSPOSED!!
        if self.config.sparsify_columns:
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

        bold_idx = 'grt-bold' if self.config.font_bold_index else ''
        html.append("<tbody>")
        for i, (n, r) in enumerate(self.df_html.iterrows()):
            # one per row of dataframe
            html.append("<tr>")
            hrule = ''
            if self.show_index:
                for j, c in enumerate(r.iloc[:self.nindex]):
                    # dx = data in index
                    # if this is the level that changes for this row
                    # will use a top rule  hence omit i = 0 which already has an hrule
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
        self._df_html_text = Escaping.clean_html_tex(text)
        logger.info('CREATED HTML')
        self._df_style_text = self.make_style(tabs)

    def clean_style(self, soup):
        """Minify CSS inside <style> blocks and remove slash-star comments."""
        if not self.config.debug:
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
            if self._df_html_text == '':
                # makes style and html (need tabs)
                self.make_html()
            code = ["<div class='greater-table'>",
                    self._df_style_text,
                    self._df_html_text,
                    "</div>"]
            soup = BeautifulSoup('\n'.join(code), 'html.parser')
            soup = self.clean_style(soup)
            self._clean_html = str(soup)  # .prettify() -> too many newlines
            logger.info('CREATED COMBINED HTML and STYLE')
        return self._clean_html

    def make_tikz(self):
        """
        Write DataFrame to custom tikz matrix.

        Updated version that uses self.df and does not need to
        reapply formatters or sparsify. Various HTML->TeX replacements
        are still needed, e.g., dealing with % and _ outside formulas.

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

        Issue: column with floats and spaces or missing causes problems (VaR,
        TVaR, EPD, mean and CV table)

        From great.pres_maker.df_to_tikz

        Arguments moved into config:

          column_sep=4 / 8,   # was 3/8
          row_sep=1 / 8,
          container_env='table',
          extra_defs='',
          hrule=None,
          vrule=None,
          post_process='',
          latex=None,


        keyword args:

            scale           picks up self.config.tikz_scale; scale applied to whole
                            table - default 0.717
            height          row height, rec. 1 (em)
            column_sep      col sep in em
            row_sep         row sep in em
            container_env   table, figure or sidewaysfigure
            color           color for text boxes (helps config.debugging)
            extra_defs      TeX defintions and commands put at top of table,
                            e.g., \\centering
            lines           lines below these rows, -1 for next to last row
                            etc.; list of ints
            post_process    e.g., non-line commands put at bottom of table
            latex           arguments after \\begin{table}[latex]
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
        # pull out arguments (convert to local vars - these used to be arguments)
        column_sep = self.config.tikz_column_sep
        row_sep = self.config.tikz_row_sep
        container_env = self.config.tikz_container_env
        hrule = self.config.tikz_hrule
        vrule = self.config.tikz_vrule
        post_process = self.config.tikz_post_process
        latex = self.config.tikz_latex

        # TODO: really should just work with this not a copy?
        df = self.df_tex.copy()
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
        # centering handled by quarto
        header = """
\\begin{{{container_env}}}{latex}
{caption}
% \\centering{{
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
% }}   % close centering
\\end{{{container_env}}}
"""

        nc_index = self.nindex
        nr_columns = self.ncolumns

        if vrule is None:
            vrule = set()
        else:
            vrule = set(vrule)
        # to the left of... +1
        vrule.add(nc_index + 1)

        logger.info(
            f'rows in columns {nr_columns}, columns in index {nc_index}')

        # internal TeX code (same as HTML code)
        matrix_name = self.df_id

        # column and tikz display widths
        colw = self.tex_knowledge_df['tikz_colw'].map(lambda x: np.round(x, 3))
        tabs = self.tex_knowledge_df['recommended'].map(lambda x: np.round(x, 3))
        # these are indexed with pre-TeX mangling names
        # colw.index = df.columns
        # tabs.index = df.columns

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
        if self.config.debug:
            # color all boxes
            debug = ', draw=blue!10'
        else:
            debug = ''
        sio.write(header.format(container_env=container_env,
                                caption=caption,
                                scale=self.config.tikz_scale,
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
                f'\trow {i}/.style={{nodes={{text=black, anchor=south, inner ysep=.2em, minimum height=1.3em, font=\\bfseries, align=center}}}},\n')

        # override for index columns headers
        # probably ony need for the bottom row with a multiindex?
        for i in range(2, nr_columns + 2):
            for j in range(1, 1+nc_index):
                sio.write(
                    f'\trow {i} column {j}/.style='
                    '{nodes={font=\\bfseries\\itshape, align=left}},\n'
                )
        # write column spec
        for i, w, al in zip(range(1, len(align) + 1), tabs, align):
            # average char is only 0.48 of M
            # https://en.wikipedia.org/wiki/Em_(gtypography)
            if i == 1:
                # first column sets row height for entire row
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, '
                          'text height=0.9em, text depth=0.2em, '
                          f'inner xsep={column_sep}em, inner ysep=0, '
                          f'text width={max(2, w):.2f}em}},\n')
            else:
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, nosep, text width={max(2, w):.2f}em}},\n')
        # extra col to right which enforces row height
        sio.write(
            f'\tcolumn {i+1:>2d}/.style={{text height=0.9em, text depth=0.2em, nosep, text width=0em}}\n')
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
                sparse_columns[lvl], mi_vrules[lvl] = Sparsify.sparsify_mi(df.columns.get_level_values(lvl),
                                                                     lvl == len(df.columns.levels) - 1)
                for cn, c, al in zip(df.columns, sparse_columns[lvl], align):
                    # c = wfloat_format(c)
                    s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                    nl = '\\&'
                    sio.write(s.format(cell=c + '\\I'))
                # include the blank extra last column
                sio.write('\\& \\\\\n')
        else:
            nl = ''
            for c, al in zip(df.columns, align):
                # c = wfloat_format(c)
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=c + '\\I'))
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

        def python_2_tex(x):
            return x + nr_columns + 2 if x >= 0 else nr + x + 3

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
        if not all(df == self.df_tex):
            logger.error('In tikz and df has changed...')
        return sio.getvalue()

    def make_rich(self, console, box_style=box.SQUARE):
        """Render to a rich table using Console object console."""
        # figure col widths and aligners
        cw = self.text_knowledge_df['recommended']
        aligners = self.text_knowledge_df['alignment']
        show_lines = self.config.hrule_widths[0] > 0

        self._rich_table = table = (
            RichOutput.make_rich_table(self.df, cw, aligners, num_index_columns=self.nindex,
                             title=self.caption, show_lines=show_lines,
                             box_style=box_style))
        return table

    def make_string(self):
        """Print to string using custom (i.e., not Tabulate) functionality."""
        if self.df.empty:
            return ""
        if self._string == "":
            cw = self.text_knowledge_df['recommended']
            aligners = self.text_knowledge_df['alignment']
            self._string = TextOutput.make_text_table(
                self.df, cw, aligners, index_levels=self.nindex)
        return self._string

    def make_svg(self):
        """Render tikz into svg text."""
        tz = Etcher(self._repr_latex_(),
                    self.config.table_font_pt_size,
                    file_name=self.df_id
                    )
        p = tz.file_path.with_suffix('.svg')
        if not p.exists():
            try:
                tz.process_tikz()
            except ValueError as e:
                print(e)
                return "no svg output"

        txt = p.read_text()
        return txt

    def save_html(self, fn):
        """Save HTML to file."""
        html_boiler_plate = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Greater Table</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Web-safe fonts fallback -->
  <link href="https://fonts.googleapis.com/css2?family=Roboto&family=Open+Sans+Condensed:ital,wght@0,300;1,300&display=swap" rel="stylesheet">

  <!-- MathJax for TeX rendering -->
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body {
      font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
      margin: 2em;
      background: #fff;
      color: #000;
    }
  </style>
</head>
<body>

<h1>Rendered Table</h1>

{table_html}

</body>
</html>
'''
        p = Path(fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p = p.with_suffix('.html')
        print(p)
        html = html_boiler_plate.replace('{table_html}', self.html)
        soup = BeautifulSoup(html, 'html.parser')
        p.write_text(soup.prettify(), encoding='utf-8')
        logger.info(f'Saved to {p}')

    def show_svg(self):
        """Display svg in Jupyter."""
        svg = self.make_svg()
        if svg != 'no svg output':
            display(SVG(svg))
        else:
            print('No SVG file available (TeX compile error).')

    def show_html(self, fn=''):
        if fn == '':
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                tmp_path = Path(tmp.name)
        else:
            tmp_path = Path(fn)
        self.save_html(fn=tmp_path)
        os.startfile(tmp_path)  # native Windows way to open in default browser
        return tmp_path

    @staticmethod
    def uber_test(df, show_html=False, **kwargs):
        """
        Print various diagnostics and all the formats.

        show_html -> run show_html to display in new browser tab.
        """
        f = GT(df, **kwargs)
        display(f)
        if show_html:
            f.show_html()
        print(f)
        f.show_svg()
        display(df)
        display(f.width_report())
        print(f.make_tikz())
        return f
