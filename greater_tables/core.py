"""
Core rendering logic for GreaterTables (PyArrow Enhanced Edition).

Defines the `GT` class, which formats and renders pandas DataFrames
to HTML, plain text, or LaTeX output using a validated configuration model.

Refactored [2025-03-06] to support PyArrow-backed dataframes natively.
"""

from collections import namedtuple
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
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_string_dtype,
    is_extension_array_dtype, # Crucial for PyArrow detection
    is_numeric_dtype
)
from pydantic import ValidationError
from rich import box
from IPython.display import display, SVG

from . enums import Breakability
from . config import Configurator
from . hasher import df_short_hash
from . etcher import Etcher
from . utilities import *

# Modern pandas settings
pd.set_option('future.no_silent_downcasting', True)
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class GT(object):
    """
    Create a greater_tables formatting object.

    Provides html and latex output in quarto/Jupyter accessible manner.
    Wraps AND COPIES the dataframe df. WILL NOT REFLECT CHANGES TO DF.

    **PyArrow / 2.0 Architecture Note:**
    This class now uses an "Inspect-then-Dispatch" philosophy. It does not
    coerce inputs to floats. It respects int64[pyarrow] and string[pyarrow]
    types natively, handling nulls via validity masks rather than NaN-casting.
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
        config: Configurator | None = None,
        config_path: Path | None = None,
        **overrides,
    ):
        # --- Config Loading ---
        if config and config_path:
            raise ValueError("Pass either 'config' or 'config_path', not both.")

        if config:
            base_config = config
        elif config_path:
            config_path = Path(config_path)
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                base_config = Configurator.model_validate(raw)
            except (ValidationError, OSError) as e:
                raise ValueError(f"Failed to load config from {config_path}") from e
        else:
            base_config = Configurator()

        merged = base_config.model_dump() | overrides
        self.config = Configurator(**merged)

        # --- Data Ingestion ---
        if df is None:
            df = pd.DataFrame([])
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, list):
            df = pd.DataFrame(df)
            show_index = False
            if self.config.header_row:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
        elif isinstance(df, str):
            df = df.strip()
            if df == '':
                df = pd.DataFrame([])
            else:
                df, aligners, caption, label = MD2DF.md_to_df(df)
                show_index = False
        elif GT._is_namedtuple_instance(df):
            df = GT._ntdf(df)
        else:
            raise ValueError('df must be a DataFrame, a list of lists, or a markdown table string')

        if len(df) > self.config.large_warning and not self.config.large_ok:
            raise ValueError(
                f'Large dataframe (>{self.config.large_warning} rows). Set large_ok=True.')

        if not df.columns.is_unique:
            raise ValueError('df column names are not unique')

        if caption != '':
            self.caption = caption
        else:
            self.caption = getattr(df, 'gt_caption', '')
        self.label = label

        # --- PHASE 1: Data Preparation (Modified for Arrow) ---
        self.df = df.copy()
        self.raw_df = df.copy()
        self.df_id = df_short_hash(self.df)

        if self.caption != '' and self.config.debug:
            self.caption += f' (id: {self.df_id})'

        self.show_index = show_index
        self.nindex = self.df.index.nlevels if self.show_index else 0
        self.ncolumns = self.df.columns.nlevels
        self.ncols = self.df.shape[1]
        self.dt = self.df.dtypes

        # Handle Index
        with warnings.catch_warnings():
            if self.show_index:
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                self.df = self.df.reset_index(drop=False, col_level=self.df.columns.nlevels - 1)
            # Ensure index is essentially a row number for internal tracking
            self.df.index = np.arange(self.df.shape[0], dtype=int)

        self.index_change_level = Indexing.changed_column(self.df.iloc[:, :self.nindex])
        if self.ncolumns > 1:
            self.index_change_level = pd.Series([i[-1] for i in self.index_change_level])

        self.column_change_level = Indexing.changed_level(self.raw_df.columns)

        # --- Column Tagging ---
        # Helper to process column lists/regex
        def resolve_cols(cols):
            if cols is None: return []
            if cols == 'all': return list(self.df.columns)
            if not isinstance(cols, (tuple, list)): return self.cols_from_regex(cols)
            return cols

        # Check for non-unique collision first
        if not self.df.columns.is_unique:
             logger.warning('Cols specified with non-unique column names: ignoring request.')
             self.ratio_cols, self.year_cols, self.date_cols, self.raw_cols = [], [], [], []
        else:
            self.ratio_cols = resolve_cols(ratio_cols)
            self.year_cols = resolve_cols(year_cols)
            self.date_cols = resolve_cols(date_cols)
            self.raw_cols = resolve_cols(raw_cols)

        # --- REMOVED: Aggressive Float Coercion Loop ---
        # The previous version looped through all columns and tried to cast them
        # to floats to detect types. This broke PyArrow strings/timestamps.
        # We now trust the dtypes provided by the dataframe.

        # --- Type Detection for Breakability ---
        if unbreakable is None: unbreakable = []
        elif isinstance(unbreakable, str): unbreakable = [unbreakable]

        self.float_col_indices = []
        self.integer_col_indices = []
        self.date_col_indices = []
        self.object_col_indices = []
        self.break_penalties = []

        logger.debug('FIGURING TYPES (Arrow-Aware)')
        for i, cn in enumerate(self.df.columns):
            # We look at the actual series to determine type
            ser = self.df.iloc[:, i]
            dtype = ser.dtype

            # Use pandas.api.types for robust checking across backends
            is_date = (cn in self.date_cols) or is_datetime64_any_dtype(dtype)
            is_int = is_integer_dtype(dtype)
            is_flt = is_float_dtype(dtype)

            if is_date:
                self.date_col_indices.append(i)
                self.break_penalties.append(
                    Breakability.NEVER if cn in unbreakable else Breakability.DATE)
            elif is_int:
                self.integer_col_indices.append(i)
                self.break_penalties.append(Breakability.NEVER)
            elif is_flt:
                self.float_col_indices.append(i)
                self.break_penalties.append(Breakability.NEVER)
            else:
                # String / Object / Boolean
                self.object_col_indices.append(i)
                if cn in self.year_cols or cn in self.ratio_cols:
                    self.break_penalties.append(Breakability.NEVER)
                else:
                    self.break_penalties.append(
                        Breakability.NEVER if cn in unbreakable else Breakability.ACCEPTABLE)

        # --- Alignment Logic ---
        if aligners is not None and np.any(self.df.columns.duplicated()):
            logger.warning('aligners specified with non-unique column names: ignoring request.')
            aligners = None

        if aligners is None: aligners = []
        elif isinstance(aligners, str):
            aligners = {c: a for c, a in zip(self.df.columns, aligners)}

        self.df_aligners = []
        lrc = {'l': 'grt-left', 'r': 'grt-right', 'c': 'grt-center'}

        for i, c in enumerate(self.df.columns):
            if c in aligners:
                self.df_aligners.append(lrc.get(aligners[c], 'grt-center'))
            elif i < self.nindex:
                self.df_aligners.append('grt-left')
            elif c in self.year_cols:
                self.df_aligners.append('grt-center')
            elif c in self.raw_cols:
                self.df_aligners.append('grt-left')
            elif i in self.date_col_indices:
                self.df_aligners.append('grt-center')
            elif c in self.ratio_cols or i in self.float_col_indices or i in self.integer_col_indices:
                self.df_aligners.append('grt-right')
            else:
                self.df_aligners.append('grt-left')

        self.df_idx_aligners = self.df_aligners[:self.nindex]

        # --- Formatter Setup ---
        self.user_formatters_override = {}
        if formatters:
            if callable(formatters):
                for k in self.df.columns:
                    self.user_formatters_override[k] = formatters
            else:
                for k, v in formatters.items():
                    if callable(v): self.user_formatters_override[k] = v
                    elif isinstance(v, str): self.user_formatters_override[k] = lambda x: v.format(x=x)
                    elif isinstance(v, int):
                        fmt = f'{{x:.{v}f}}'
                        self.user_formatters_override[k] = lambda x: fmt.format(x=x)
                    else:
                        raise ValueError('Formatters must be dict of callables, ints, or strings')

        # --- Tabs ---
        if tabs is None:
            self.tabs = None
        elif isinstance(tabs, (int, float)):
            self.tabs = (tabs,) * (self.nindex + self.ncols)
        elif isinstance(tabs, (np.ndarray, pd.Series, list, tuple)):
             self.tabs = tabs if len(tabs) == self.nindex + self.ncols else None
        else:
            self.tabs = None

        # --- Padding / Config ---
        if self.config.padding_trbl is not None:
            padding_trbl = self.config.padding_trbl
        else:
            spacing_map = {'tight': (0,5,0,5), 'medium': (2,10,2,10), 'wide': (4,15,4,15)}
            padding_trbl = spacing_map.get(self.config.spacing, (2,10,2,10))
        self.padt, self.padr, self.padb, self.padl = padding_trbl

        self.max_table_width_em = self.config.max_table_inch_width * 72 / self.config.table_font_pt_size

        # --- Initialization State ---
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

        # --- Apply Formatters (THE BIG CHANGE) ---
        # We keep the raw dataframe mostly untouched.
        # self.df becomes the Formatted DataFrame (Strings).
        self.df_pre_applying_formatters = self.df.copy()

        # Apply formatters modifies self.df to contain strings
        self.df = self.apply_formatters(self.df)

        # Sparsify
        if self.config.sparsify and self.nindex > 1:
            self.df = Sparsify.sparsify(self.df, self.df.columns[:self.nindex])

        # LaTeX / HTML mapping
        if self.config.tex_to_html is not None:
            self.df_html = self.df.map(self.config.tex_to_html)
        else:
            self.df_html = self.df

        if self.config.tikz_escape_tex:
            self.df_tex = Escaping.escape_df_tex(self.df)
        else:
            self.df_tex = self.df

    def __repr__(self):
        return f"GT(df_id={self.df_id})"

    def __str__(self):
        return self.make_string()

    def _repr_html_(self):
        return self.html

    def _repr_latex_(self):
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
        pattern = re.compile(regex)
        matching_cols = [
            col for col in self.df.columns
            if any(pattern.search(str(level))
                for level in (col if isinstance(col, tuple) else (col,)))
        ]
        return matching_cols

    # --- New Robust Formatters (Arrow-Safe) ---

    def _fmt_int_safe(self, x):
        """Handle Integers (PyArrow or NumPy) safely with nulls."""
        if pd.isna(x): return ""
        try:
            return self.config.default_integer_str.format(x=int(x))
        except (ValueError, TypeError):
            return str(x)

    def _fmt_date_iso(self, x):
        """Handle Dates/Timestamps safely."""
        if pd.isna(x): return ""
        # If it's a Timestamp object (Arrow or Pandas), it has strftime
        if hasattr(x, "strftime"):
            return x.strftime(self.config.default_date_str)
        return str(x)

    def _fmt_float_smart(self, x):
        """Smart float formatting that handles 'Year-like' floats."""
        if pd.isna(x): return ""
        try:
            # Check if it's effectively an integer (e.g. 2021.0)
            # This preserves the "Float as Int" behavior for legacy data
            if isinstance(x, float) and x.is_integer():
                 return self.config.default_integer_str.format(x=int(x))
            return self.config.default_float_str.format(x=x)
        except (ValueError, TypeError):
            return str(x)

    def _fmt_ratio(self, x):
        if pd.isna(x): return ""
        try: return self.config.default_ratio_str.format(x=x)
        except: return str(x)

    def _fmt_year(self, x):
        if pd.isna(x): return ""
        try: return f'{int(x):d}'
        except: return str(x)

    def _fmt_raw(self, x):
        if pd.isna(x): return ""
        return str(x)

    # --- Legacy Float Formatter Factory (Preserved but safe) ---
    def pef(self, x):
        if self._pef is None:
            self._pef = pd.io.formats.format.EngFormatter(
                accuracy=self.config.pef_precision, use_eng_prefix=True)
        return self._pef(x)

    def make_float_formatter(self, ser):
        """
        Create a customized float formatter based on column statistics.
        Works with Arrow columns as .mean(), .abs() dispatch correctly.
        """
        try:
            # Drop nulls for stats calculation to avoid issues
            ser_valid = ser.dropna()
            if len(ser_valid) == 0:
                return self._fmt_float_smart

            amean = ser_valid.abs().mean()
            # amn = ser_valid.abs().min()
            # amx = ser_valid.abs().max()

            pl, pu = 10. ** self.config.pef_lower, 10. ** self.config.pef_upper

            if amean < 1: precision = 5
            elif amean < 10: precision = 3
            elif amean < 20000: precision = 2
            else: precision = 0

            fmt = f'{{x:,.{precision}f}}'

            def ff(x):
                if pd.isna(x): return ""
                try:
                    # Check for Engineering Format conditions
                    val_abs = abs(x)
                    if (val_abs > 0) and (val_abs < pl or val_abs > pu):
                         return self.pef(x)
                    return fmt.format(x=x)
                except (ValueError, TypeError, InvalidOperation):
                    return str(x)
            return ff
        except Exception as e:
            logger.debug(f"Float formatter factory failed: {e}. Using default.")
            return self._fmt_float_smart

    @property
    def df_formatters(self):
        """
        Dispatcher: Inspects types and assigns formatters.
        """
        if self._df_formatters is None:
            self._df_formatters = []

            # Pre-calc custom table-wide float format if it exists
            custom_float = None
            if self.config.table_float_format:
                if callable(self.config.table_float_format):
                    # Wrap to handle safe calls
                    def safe_custom_float(x):
                        try:
                            return self.config.table_float_format(x=x)
                        except ValueError:
                            return str(x)
                        except Exception as e:
                            logger.error(f'Custom float function raised {e=}')
                            return str(x)
                    custom_float = safe_custom_float
                else:
                    fmt = self.config.table_float_format
                    def safe_custom_float_str(x):
                        try:
                            return fmt.format(x=x)
                        except ValueError:
                            return str(x)
                        except Exception as e:
                            logger.error(f'Custom float format string raised {e=}')
                            return str(x)
                    custom_float = safe_custom_float_str

            for i, col_name in enumerate(self.df.columns):
                # 1. User overrides (highest priority)
                if col_name in self.user_formatters_override:
                    self._df_formatters.append(self.user_formatters_override[col_name])
                    continue

                # 2. Semantic Tags
                if col_name in self.ratio_cols:
                    self._df_formatters.append(self._fmt_ratio)
                    continue
                if col_name in self.year_cols:
                    self._df_formatters.append(self._fmt_year)
                    continue
                if col_name in self.raw_cols:
                    self._df_formatters.append(self._fmt_raw)
                    continue

                # 3. Type-Based Dispatch (The "PyArrow" logic)
                dtype = self.df[col_name].dtype

                # Date/Time
                if (i in self.date_col_indices) or is_datetime64_any_dtype(dtype):
                    self._df_formatters.append(self._fmt_date_iso)

                # Integer (NumPy or Arrow)
                elif is_integer_dtype(dtype):
                    self._df_formatters.append(self._fmt_int_safe)

                # Float (NumPy or Arrow)
                elif is_float_dtype(dtype):
                    if custom_float:
                        self._df_formatters.append(custom_float)
                    else:
                        # Use the smart factory
                        self._df_formatters.append(self.make_float_formatter(self.df.iloc[:, i]))

                # Default / String / Object
                else:
                    self._df_formatters.append(self._fmt_raw)

            if len(self._df_formatters) != self.df.shape[1]:
                raise ValueError(f'Formatter count mismatch: {len(self._df_formatters)} != {self.df.shape[1]}')

        return self._df_formatters

    @staticmethod
    def apply_formatters_work(df, formatters):
        """Apply formatters to a DataFrame."""
        try:
            # This applies the lambda functions to every cell
            new_df = pd.DataFrame({
                i: map(f, df.iloc[:, i])
                for i, f in enumerate(formatters)
            }, index=df.index) # Preserve index!
        except TypeError:
            print('NASTY TYPE ERROR')
            raise

        new_df.columns = df.columns

        # OPTIMIZATION: Convert to PyArrow strings immediately.
        # This enables vectorized width calculations later.
        try:
            return new_df.astype("string[pyarrow]")
        except ImportError:
            # Fallback if pyarrow not installed (unlikely given context)
            return new_df.astype(str)

    def apply_formatters(self, df, mode='adjusted'):
        """
        Replace df (the raw df) with formatted string df.
        """
        if mode == 'adjusted':
            return GT.apply_formatters_work(df, self.df_formatters)
        elif mode == 'raw':
            data_formatters = self.df_formatters[self.nindex:]
            new_body = GT.apply_formatters_work(df, data_formatters)
            if not self.show_index:
                return new_body

            index_formatters = self.df_formatters[:self.nindex]
            df_index = df.reset_index(
                drop=False, col_level=self.df.columns.nlevels - 1).iloc[:, :self.nindex]
            new_index = GT.apply_formatters_work(df_index, index_formatters)

            new_df = pd.concat([new_index, new_body], axis=1)
            new_df = new_df.set_index(list(df_index.columns))
            new_df.index.names = df.index.names
            return new_df
        else:
            raise ValueError(f'unknown mode {mode}')

    # --- Knowledge DFs ---
    @property
    def text_knowledge_df(self):
        if self._text_knowledge_df is None:
            self._text_knowledge_df = self.estimate_column_widths_by_mode('text')
        return self._text_knowledge_df

    @property
    def html_knowledge_df(self):
        if self._html_knowledge_df is None:
            self._html_knowledge_df = self.estimate_column_widths_by_mode('html')
        return self._html_knowledge_df

    @property
    def tex_knowledge_df(self):
        if self._tex_knowledge_df is None:
            if not self.config.tikz:
                self._tex_knowledge_df = self.html_knowledge_df
            else:
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
        """
        Return dataframe of width information.
        OPTIMIZED: Uses vectorized string operations for 'text' mode.
        """
        assert mode in ('text', 'html', 'tex'), 'Only html, text and tex modes valid.'

        if mode == 'text':
            df = self.df # This is now string[pyarrow] thanks to apply_formatters
            # Vectorized length calculation (Much faster than map(len))
            # We assume df is already string[pyarrow]
            try:
                # Ideally: df.apply(lambda x: x.str.len().max())
                # Since df is objects/strings, .str accessor works if dtype is string
                if is_string_dtype(df.iloc[:,0]):
                     natural_width = df.apply(lambda x: x.str.len().max()).to_dict()
                else:
                     natural_width = df.map(len).max(axis=0).to_dict()
            except:
                 natural_width = df.map(len).max(axis=0).to_dict()

            len_function = len
            bold_adjustment = 1.0
        elif mode == 'html':
            df = self.df_html
            len_function = TextLength.text_display_len
            bold_adjustment = 1.1
            natural_width = df.map(lambda x: len_function(x.strip())).max(axis=0).to_dict()
        else:
            df = self.df_tex
            len_function = TextLength.text_display_len
            bold_adjustment = 1.1
            natural_width = df.map(lambda x: len_function(x.strip())).max(axis=0).to_dict()

        n_row, n_col = df.shape

        # in text mode: figure out where you can break; pat breaks after punctuation or at -
        pat = r'(?<=[.,;:!?)\]}\u2014\u2013])\s+|--*\s+|\s+'
        iso_date_split = r'(?<=\b\d{4})-(?=\d{2}-\d{2})'
        pat = f'{pat}|{iso_date_split}'

        minimum_width = {}
        header_natural = {}
        header_minimum = {}

        for col_name in df.columns:
            # For minimum width, we still need splitting
            minimum_width[col_name] = (
                df[col_name].astype(str).str # Ensure str accessor
                .split(pat=pat, regex=True, expand=True)
                .fillna('')
                .map(len_function)
                .max(axis=1)
                .max()
            )

            ctuple = col_name if isinstance(col_name, tuple) else (col_name, )
            header_natural[col_name] = bold_adjustment * max(map(len_function, ctuple))
            header_minimum[col_name] = bold_adjustment * min(len_function(part) for i in ctuple for part in re.split(pat, str(i)))

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
            ans['natural_width'] += 1
            ans['minimum_width'] += 1
            ans['header_natural'] += 1
            ans['header_minimum'] += 1

        natural, acceptable, minimum = ans.iloc[:, 3:6].sum()
        head_natural, head_minimum = ans.iloc[:, 6:8].sum()

        if mode == 'text':
            PADDING = 2
            pad_adjustment = (PADDING + 1) * n_col - 1
        else:
            PADDING = 1
            pad_adjustment =  PADDING * n_col

        if self.config.table_width_mode == 'explicit':
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

        if self.config.table_width_header_adjust > 0:
            max_extra = int(self.config.table_width_header_adjust * target_width)
        else:
            max_extra = 0

        if target_width > natural:
            ans['recommended'] = ans['natural_width']
            space = target_width - natural
            logger.info('Space for NATURAL! Spare space = %s', space)
        elif target_width > acceptable:
            ans['recommended'] = ans['acceptable_width']
            space = target_width - acceptable
            logger.info('Using "breaks acceptable" (dates not wrapped), spare space = %s', space)
        elif target_width > minimum:
            ans['recommended'] = ans['minimum_width']
            space = target_width - minimum
            logger.info('Using "minimum" (all breakable incl dates), spare space = %s', space)
        else:
            ans['recommended'] = ans['minimum_width']
            space = target_width - minimum
            logger.info('Mode %s, desired width too small, table too wide by %s em.', mode, space)

        logger.info(f'{mode=} {target_width=}, {natural=}, {acceptable=}, {minimum=}, {max_extra=}, {space=}')

        if mode == "text" and space > 0 and df.columns.nlevels == 1:
            ans['raw_recommended'] = ans['recommended']
            if max_extra > 0:
                adj = Width.header_adjustment(df, ans['recommended'], space, max_extra)
                ans['header_tweak'] = pd.Series(adj)
            else:
                ans['header_tweak'] = 0
            ans['recommended'] = ans['recommended'] + ans['header_tweak']
            ans['header_natural'] = ans['recommended']
            ans['header_minimum'] = ans['recommended']

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
            tikz_colw = dict.fromkeys(df.columns, 0)
            tikz_headw = dict.fromkeys(df.columns, 0)
            for i, c in enumerate(df.columns):
                c0 = c
                if not isinstance(c, tuple): c = (c,)
                c = [str(i) for i in c]
                tikz_headw[c0] = max(map(len, c))
                tikz_colw[c0] = df.iloc[:, i].map(lambda x: len(str(x))).max()
            for c in df.columns:
                tikz_colw[c] = max(tikz_colw[c], tikz_headw[c])
            ans['tikz_colw'] = tikz_colw
            ans['tikz_colw'] += 2

        return_columns = [
            'alignment', 'break_penalties', 'breakability', 'natural_width',
            'acceptable_width', 'minimum_width', 'header_natural', 'header_minimum',
            'raw_recommended', 'header_tweak', 'pre_space_share_recommended',
            'proto_recommended', 'recommended', 'tikz_colw',
            ]
        ans = ans[[i for i in return_columns if i in ans.columns]]
        ans['recommended'] = np.maximum(ans['recommended'], 1)
        return ans

    def make_style(self, tabs):
        """Write out custom CSS for the table."""
        if self.config.debug:
            head_tb, body_b = '#0ff', '#f0f'
            h0, h1, h2 = '#f00', '#b00', '#900'
            bh0, bh1, v0, v1, v2 = '#f00', '#b00', '#0f0', '#0a0', '#090'
        else:
            head_tb = body_b = h0 = h1 = h2 = bh0 = bh1 = v0 = v1 = v2 = '#000'

        table_hrule = self.config.table_hrule_width
        table_vrule = self.config.table_vrule_width
        padt, padr, padb, padl = self.padt, self.padr, self.padb, self.padl

        style = [f'''
<style>
    #{self.df_id} {{
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
        style.append('</style>')
        logger.info('CREATED CSS')
        return '\n'.join(style)

    def make_html(self):
        """Convert a pandas DataFrame to an HTML table."""
        index_name_to_level = dict(
            zip(self.raw_df.index.names, range(self.nindex)))
        index_change_level = self.index_change_level.map(index_name_to_level)
        column_change_level = self.column_change_level

        html = [f'<table id="{self.df_id}">']
        if self.caption != '':
            html.append(f'<caption>{self.caption}</caption>')

        bit = self.df_html.T.reset_index(drop=False, allow_duplicates=True)
        idx_header = bit.iloc[:self.nindex, :self.ncolumns]
        columns = bit.iloc[self.nindex:, :self.ncolumns]

        tabs = self.html_knowledge_df['recommended'].map(lambda x: np.round(x, 3))
        tabs = np.array(tabs) + (self.padl + self.padr) / 12

        html.append('<colgroup>')
        for w in tabs:
            html.append(f'<col style="width: {w}em;">')
        html.append('</colgroup>')

        if self.config.sparsify_columns:
            html.append("<thead>")
            for i in range(self.ncolumns):
                html.append("<tr>")
                if self.show_index:
                    for j, r in enumerate(idx_header.iloc[:, i]):
                        html.append(f'<th class="grt-left">{r}</th>')
                cum_col = 0
                for j, (nm, g) in enumerate(groupby(columns.iloc[:, :i + 1].
                                                    apply(lambda x: ':::'.join(str(i) for i in x), axis=1))):
                    nm = nm.split(':::')[-1]
                    hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
                    colspan = sum(1 for _ in g)
                    if 0 < j:
                        vrule = f'grt-vrule-{column_change_level[cum_col]}'
                    elif j == 0 and self.show_index:
                        vrule = f'grt-vrule-index'
                    else:
                        vrule = ''
                    if j == 0 and not self.show_index:
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
                html.append("<tr>")
                if self.show_index:
                    for j, r in enumerate(idx_header.iloc[:, i]):
                        html.append(f'<th class="grt-left">{r}</th>')
                for j, r in enumerate(columns.iloc[:, i]):
                    hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
                    if 0 < j < self.ncols and i >= column_change_level[j]:
                        vrule = f'grt-vrule-{column_change_level[j]}'
                    elif j == 0 and self.show_index:
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
            html.append("<tr>")
            hrule = ''
            if self.show_index:
                for j, c in enumerate(r.iloc[:self.nindex]):
                    if i > 0 and hrule == '' and i in index_change_level and j == index_change_level[i]:
                        hrule = f'grt-hrule-{j}'
                    col_id = f'grt-c-{j}'
                    html.append(
                        f'<td class="{col_id} {bold_idx} {self.df_aligners[j]} {hrule}">{c}</td>')
            for j, c in enumerate(r.iloc[self.nindex:]):
                if 0 < j < self.ncols:
                    vrule = f'grt-vrule-{column_change_level[j]}'
                elif j == 0 and self.show_index:
                    vrule = f'grt-vrule-index'
                else:
                    vrule = ''
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
                    cleaned_css = re.sub(r'/\*.*?\*/', '', style_tag.string, flags=re.DOTALL)
                    style_tag.string.replace_with(cleaned_css)
        return soup

    @property
    def html(self):
        if self._clean_html == '':
            if self._df_html_text == '':
                self.make_html()
            code = ["<div class='greater-table'>", self._df_style_text, self._df_html_text, "</div>"]
            soup = BeautifulSoup('\n'.join(code), 'html.parser')
            soup = self.clean_style(soup)
            self._clean_html = str(soup)
            logger.info('CREATED COMBINED HTML and STYLE')
        return self._clean_html

    def make_tikz(self):
        """
        Write DataFrame to custom tikz matrix.
        """
        if not self.config.tikz:
            return ''
        column_sep = self.config.tikz_column_sep
        row_sep = self.config.tikz_row_sep
        container_env = self.config.tikz_container_env
        hrule = self.config.tikz_hrule
        vrule = self.config.tikz_vrule
        post_process = self.config.tikz_post_process
        latex = self.config.tikz_latex

        df = self.df_tex.copy()
        caption = self.caption
        label = self.label
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
            raise ValueError('tikz routine requires unique column names')

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
        vrule.add(nc_index + 1)

        logger.info(
            f'rows in columns {nr_columns}, columns in index {nc_index}')

        matrix_name = self.df_id
        colw = self.tex_knowledge_df['tikz_colw'].fillna(0).round(3)
        tabs = self.tex_knowledge_df['recommended'].map(lambda x: np.round(x, 3))

        ad = {'l': 'left', 'r': 'right', 'c': 'center'}
        ad2 = {'l': '<', 'r': '>', 'c': '^'}
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

        sio = StringIO()
        if latex is None:
            latex = ''
        else:
            latex = f'[{latex}]'
        if self.config.debug:
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

        i = 1
        sio.write(
            f'\trow {i}/.style={{nodes={{text=black, anchor=north, inner ysep=0, text height=0, text depth=0}}}},\n')
        for i in range(2, nr_columns + 2):
            sio.write(
                f'\trow {i}/.style={{nodes={{text=black, anchor=south, inner ysep=.2em, minimum height=1.3em, font=\\bfseries, align=center}}}},\n')

        for i in range(2, nr_columns + 2):
            for j in range(1, 1+nc_index):
                sio.write(
                    f'\trow {i} column {j}/.style='
                    '{nodes={font=\\bfseries\\itshape, align=left}},\n'
                )
        for i, w, al in zip(range(1, len(align) + 1), tabs, align):
            if i == 1:
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, '
                          'text height=0.9em, text depth=0.2em, '
                          f'inner xsep={column_sep}em, inner ysep=0, '
                          f'text width={max(2, w):.2f}em}},\n')
            else:
                sio.write(f'\tcolumn {i:>2d}/.style={{'
                          f'nodes={{align={ad[al]:<6s}}}, nosep, text width={max(2, w):.2f}em}},\n')
        sio.write(
            f'\tcolumn {i+1:>2d}/.style={{text height=0.9em, text depth=0.2em, nosep, text width=0em}}\n')
        sio.write('\t}]\n')

        sio.write("\\matrix ({matrix_name}) [table, ampersand replacement=\\&]{{\n".format(
            matrix_name=matrix_name))

        nl = ''
        for cn, al in zip(df.columns, align):
            s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
            nl = '\\&'
            sio.write(s.format(cell=' '))
        sio.write('\\& \\\\\n')

        mi_vrules = {}
        sparse_columns = {}
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in range(len(df.columns.levels)):
                nl = ''
                sparse_columns[lvl], mi_vrules[lvl] = Sparsify.sparsify_mi(df.columns.get_level_values(lvl),
                                                                     lvl == len(df.columns.levels) - 1)
                for cn, c, al in zip(df.columns, sparse_columns[lvl], align):
                    s = f'{nl} {{cell:{ad2[al]}{colw[cn]}s}} '
                    nl = '\\&'
                    sio.write(s.format(cell=c + '\\I'))
                sio.write('\\& \\\\\n')
        else:
            nl = ''
            for c, al in zip(df.columns, align):
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=str(c) + '\\I'))
            sio.write('\\& \\\\\n')

        for idx, row in df.iterrows():
            nl = ''
            for c, cell, al in zip(df.columns, row, align):
                s = f'{nl} {{cell:{ad2[al]}{colw[c]}s}} '
                nl = '\\&'
                sio.write(s.format(cell=cell))
            sio.write('\\& \\\\\n')
        sio.write(f'}};\n\n')

        nr, nc = df.shape
        nr += nr_columns + 1

        def python_2_tex(x):
            return x + nr_columns + 2 if x >= 0 else nr + x + 3

        tb_rules = [nr_columns + 1, python_2_tex(-1)]
        if hrule:
            hrule = set(map(python_2_tex, hrule)).union(tb_rules)
        else:
            hrule = list(tb_rules)
        logger.debug(f'hlines: {hrule}')

        yshift = row_sep / 2
        xshift = -column_sep / 2
        descender_proportion = 0.25

        ls = 'thick'
        ln = 1
        sio.write(
            f'\\path[draw, {ls}] ({matrix_name}-{ln}-1.south west)  -- ({matrix_name}-{ln}-{nc+1}.south east);\n')

        for ln in hrule:
            ls = 'thick' if ln == nr + nr_columns + \
                1 else ('semithick' if ln == 1 + nr_columns else 'very thin')
            if ln < nr:
                sio.write(f'\\path[draw, {ls}] ([yshift={-yshift}em]{matrix_name}-{ln}-1.south west)  -- '
                          f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')
            else:
                ln = nr
                sio.write(f'\\path[draw, thick] ([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-1.base west)  -- '
                          f'([yshift={-descender_proportion-yshift}em]{matrix_name}-{ln}-{nc+1}.base east);\n')

        if nr_columns > 1:
            for ln in range(2, nr_columns + 1):
                sio.write(f'\\path[draw, very thin] ([xshift={xshift}em, yshift={-yshift}em]'
                          f'{matrix_name}-{ln}-{nc_index+1}.south west)  -- '
                          f'([yshift={-yshift}em]{matrix_name}-{ln}-{nc+1}.south east);\n')

        written = set(range(1, nc_index + 1))
        if vrule and self.show_index:
            ls = 'very thin'
            for cn in vrule:
                if cn not in written:
                    sio.write(f'\\path[draw, {ls}] ([xshift={xshift}em]{matrix_name}-1-{cn}.south west)  -- '
                              f'([yshift={-descender_proportion-yshift}em, xshift={xshift}em]{matrix_name}-{nr}-{cn}.base west);\n')
                    written.add(cn - 1)

        if len(mi_vrules) > 0:
            logger.debug(
                f'Generated vlines {mi_vrules}; already written {written}')
            ls = 'ultra thin'
            for k, cols in mi_vrules.items():
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

  <link href="https://fonts.googleapis.com/css2?family=Roboto&family=Open+Sans+Condensed:ital,wght@0,300;1,300&display=swap" rel="stylesheet">

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

    @staticmethod
    def _is_namedtuple_instance(x) -> bool:
        """Heuristic: namedtuple instances are tuples whose class defines _fields."""
        return isinstance(x, tuple) and isinstance(getattr(type(x), "_fields", None), tuple)

    @staticmethod
    def _ntdf(t):
        """Convert named tuple to pandas dataframe to display."""
        return pd.Series(t, index=pd.Index(t._fields, name="Item")).to_frame('Value')
