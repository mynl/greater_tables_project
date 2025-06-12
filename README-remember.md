

Here's how you subclass.


```python
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
        table_vrule_width = 1 if ncolumns == 1 else (
            1.5 if ncolumns == 2 else 2)

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

```
