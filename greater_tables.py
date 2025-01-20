# table formatting again
import pandas as pd
# from IPython.display import HTML


def gtqd(df, col_align='', formatters=None, ratio_cols=None, **kwargs):
    """Better HTML output for a DataFrame."""
    table_id = f'T{id(df):x}'[::2].upper()
    style = []

    def r(x):
        """Default ratio format."""
        try:
            return f'{x:.1%}'
        except:
            return x

    def _float(x):
        try:
            return f'{x:,.3f}'
        except:
            return x

    def _int(x):
        try:
            return f'{x:,.3f}'
        except:
            return x

    # see if index col names in formatters?
    dt = df.dtypes
    if ratio_cols is None:
        ratio_cols = []
    elif not isinstance(ratio_cols, (list, tuple)):
        ratio_cols = list(ratio_cols)

    float_cols = df.select_dtypes(include='float').columns
    integer_cols = df.select_dtypes(include='int').columns
    if formatters is None:
        formatters = {}

    for c in df.columns:
        if c not in formatters.keys():
            # set a default
            formatters[c] = r if c in ratio_cols else (
                _float if c in float_cols else (
                    _int if c in integer_cols else
                    lambda x: x)
                )

    html = df.to_html(table_id=table_id, formatters=formatters, **kwargs)

    if col_align == '':
        # guess: index l, numeric r rest l
        idx = 'l' * df.index.nlevels
        numeric_cols = df.select_dtypes('number').columns
        rc = ''.join('r' if c in numeric_cols else 'l' for c in df.columns)
        col_align = idx + rc

    # col no -> lrc -> spell out
    d = {'l': 'left', 'r': 'right', 'c':'center'}
    ca = col_align
    ca = dict(zip(range(1, 1+len(ca)), map(d.get, ca)))
    style.append('<style>')

    style.append(f'''
#{table_id}  {{
/*  border-collapse: collapse;*/
  width: 100%;
/*  margin above left below right table*/
  margin: 0px 0 10px 0;
  font-family: "Open Sans Condensed", "Arial Narrow", Arial, "Roboto Condensed",  sans-serif;
  font-size: 0.9em;
}}

''')


    for i in range(1, 1+len(ca)):
        style.append(
            f'#{table_id} tbody td:nth-child({i}){{ text-align: {ca[i]}; }}')
    style.append(f'#{table_id} th {{ text-align: center;}}')
    style.append('</style>\n')

    if len(style):
        style = '\n'.join(style)
    else:
        style = ''

    out = f'{style}{html}'

    return out


# def pf(df, *, ratio_cols=None, precision=3, pef_lower=-3, pef_upper=16,
#     format_index=True):
#     """Format a DataFrame."""

#     df = df.copy()

#     _ratio_names = ['max_LR', 'gross_LR', 'net_LR', 'ceded_LR', 'LR',
#                     'COC']

#     if ratio_cols == 'all':
#         ratio_cols = [i for i in df.columns]

#     elif ratio_cols is not None and type(ratio_cols) != list:
#         ratio_cols = [ratio_cols]

#     def pef(x):
#         """Pandas engineering formatter."""
#         return pd.io.formats.format.EngFormatter(accuracy=2, use_eng_prefix=True)(x)

#     pl = 10.**pef_lower
#     pu = 10.**pef_upper

#     def nf(x):
#         """Number formatter."""
#         try:
#             if x == int(x):
#                 return f'{x:,.0f}'
#             elif abs(x - 1) < 1e-3:
#                 return f"1-{1 - x:.3g}"
#             elif abs(x) < pl or abs(x) > pu:
#                 return pef(x)
#             elif abs(x) > 1e2:
#                 fmt = f'{{x:,.{precision - 1}f}}'
#                 return fmt.format(x=x)
#                 # return f'{x:,.1f}'
#             else:
#                 fmt = f'{{x:,.{precision}f}}'
#                 return fmt.format(x=x)
#         except:
#             return x

#     def ratio(x):
#         try:
#             return f'{x:.1%}'
#         except:
#             return x

#     def integer(x):
#         return f'{x:,d}'

#     # convert into string
#     col_list = [f'{c}' for c in df.columns]

#     if ratio_cols is None:
#         ratio_cols = [c for c in col_list if c in _ratio_names]
#         if len(ratio_cols) == 0:
#             ratio_cols = None
#     if ratio_cols is not None:
#         col_list = list(set(col_list) - set(ratio_cols))

#     number_cols = df.select_dtypes(include='number').columns

#     index_cache = None
#     if format_index:
#         index_cache = df.index.names
#         df = df. reset_index(drop=False)

#     float_cols = df.select_dtypes(include='float').columns
#     integer_cols = df.select_dtypes(include='int').columns

#     for c in df:
#         # if df.dtypes[c] in (int, float)
#         if c in ratio_cols:
#             df[c] = df[c].map(ratio)
#         elif c in float_cols:
#             df[c] = df[c].map(nf)
#         elif c in integer_cols:
#             df[c] = df[c].map(integer)
#         else:
#             print(f'Col {c} not treated')

#     if format_index and index_cache is not None:
#         df = df.set_index(index_cache)

#     # align number columns
#     # method 1
#     # sdf = (
#     #     df.style
#     #     .applymap(lambda x: 'text-align: right;', subset=number_cols
#     #         )
#     # )

#     # Define styles for specific columns
#     # styles = [
#     #     {'selector': f'td.col{i}', 'props': [('text-align', 'right')]}  # Apply to specific columns
#     #     for i in number_cols
#     # ]

#     # # Apply styles
#     # styled_df = df.style.set_table_styles(styles)

#     # display(styled_df)


#     # Generate table HTML with inline CSS
#     table_html = df.to_html(index=True, classes="dataframe")


#     return HTML(table_html)
