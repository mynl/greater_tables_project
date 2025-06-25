"""
Text length, column width balancing and other utilities.
"""

import html
from io import StringIO
import logging
import re
from textwrap import wrap

import pandas as pd
from pybtex.textutils import width
from rich import box
from rich.table import Table

from . formats import GT_Format, TableFormat, Line, DataRow


__all__ = ['MD2DF', 'Escaping', 'TextLength',
           'Sparsify', 'Indexing', 'Width', 'TextOutput',
           'RichOutput']


logger = logging.getLogger(__name__)


class MD2DF:
    """Convert markdown to dataframe."""
    @staticmethod
    def md_to_df(txt):
        """Convert markdown text string table to DataFrame."""
        # extract table and optional caption part
        table, caption = MD2DF.parse_markdown_table_and_caption(txt)
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


class Escaping:
    """Escape html and tex within tables."""
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

    @staticmethod
    def clean_index(df):
        """
        escape _ for columns and index, being careful about subscripts
        in TeX formulas.

        :param df:
        :return:
        """
        return df.rename(index=Escaping.clean_name, columns=Escaping.clean_name)

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
    def escape_tex_outside_math(text):
        # Pattern to match math environments: $...$, $$...$$, \[...\]
        if not isinstance(text, str):
            return text
        math_pattern = re.compile(
            r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\])', re.DOTALL)

        # def escape_non_math(s):
        #     return s.replace('\\', r'\\').replace('%', r'\%')
        # because of use within tikz tables
        def escape_non_math(s):
            return s.replace('\\', r'\textbackslash{}').replace('%', r'\%').replace('_', r'\_')

        parts = []
        last_end = 0
        for m in math_pattern.finditer(text):
            start, end = m.span()
            parts.append(escape_non_math(text[last_end:start]))
            parts.append(m.group())  # math part, unescaped
            last_end = end
        parts.append(escape_non_math(text[last_end:]))

        return ''.join(parts)

    @staticmethod
    def escape_df_tex(df):
        # Escape data elements
        df = df.map(Escaping.escape_tex_outside_math)

        # Escape index and column values
        def escape_index(idx):
            if isinstance(idx, pd.MultiIndex):
                return pd.MultiIndex.from_tuples(
                    [tuple(Escaping.escape_tex_outside_math(x)
                           for x in tup) for tup in idx],
                    names=[Escaping.escape_tex_outside_math(
                        n) if n else n for n in idx.names]
                )
            else:
                return pd.Index([Escaping.escape_tex_outside_math(x) for x in idx],
                                name=Escaping.escape_tex_outside_math(idx.name) if idx.name else None)

        df.index = escape_index(df.index)
        df.columns = escape_index(df.columns)

        return df


class TextLength:
    """Estimate length of displayed text."""
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

    @staticmethod
    def approximate_char_width_em(c: str) -> float:
        width_table = {
            "il.':|!`": 0.3,
            "frtJ(){}[]*": 0.5,
            "abcdeghknopqsuvxyz": 0.6,
            "LCDEFHISTUZ": 0.7,
            "ABGKNOPQRXYV": 0.8,
            "mwMW": 0.9,
            "0123456789": 0.6,
            "-_=+<>": 0.5,
            "#$%^&@~": 0.6,
            ",;": 0.25,
            '"': 0.4,
            "/\\": 0.5,
            "?": 0.6,
            " ": 0.4,
        }
        width_table = {
        "a": 0.444,
        "b": 0.5,
        "c": 0.444,
        "d": 0.5,
        "e": 0.444,
        "f": 0.333,
        "g": 0.5,
        "h": 0.5,
        "i": 0.278,
        "j": 0.278,
        "k": 0.5,
        "l": 0.278,
        "m": 0.778,
        "n": 0.5,
        "o": 0.5,
        "p": 0.5,
        "q": 0.5,
        "r": 0.333,
        "s": 0.389,
        "t": 0.278,
        "u": 0.5,
        "v": 0.5,
        "w": 0.722,
        "x": 0.5,
        "y": 0.5,
        "z": 0.444,
        "A": 0.722,
        "B": 0.667,
        "C": 0.667,
        "D": 0.722,
        "E": 0.611,
        "F": 0.556,
        "G": 0.722,
        "H": 0.722,
        "I": 0.333,
        "J": 0.389,
        "K": 0.722,
        "L": 0.611,
        "M": 0.889,
        "N": 0.722,
        "O": 0.722,
        "P": 0.556,
        "Q": 0.722,
        "R": 0.667,
        "S": 0.556,
        "T": 0.611,
        "U": 0.722,
        "V": 0.722,
        "W": 0.944,
        "X": 0.722,
        "Y": 0.722,
        "Z": 0.611,
        "0": 0.5,
        "1": 0.5,
        "2": 0.5,
        "3": 0.5,
        "4": 0.5,
        "5": 0.5,
        "6": 0.5,
        "7": 0.5,
        "8": 0.5,
        "9": 0.5,
        ".": 0.25,
        ",": 0.25,
        ":": 0.278,
        ";": 0.278,
        "(": 0.333,
        ")": 0.333,
        "[": 0.333,
        "]": 0.333,
        "’": 0.333,
        '"': 0.444,
        "!": 0.333,
        "?": 0.444,
        " ": 0.25,
        "|": 0.2,
        "‘": 0.333,
        "{": 0.48,
        "}": 0.48,
        "-": 0.333,
    }
        char_width = {c: w for chars, w in width_table.items() for c in chars}
        return char_width.get(c, 0.6)

    @staticmethod
    def text_display_len(s: str) -> float:
        """Estimate display width in ems, ignoring HTML tags, interpreting TeX, and HTML entities."""
        def strip_html_tags(text):
            return re.sub(r'<[^>]*>', '', text)

        def decode_entities(text):
            return html.unescape(text)

        if '$' not in s and '<' not in s and '&' not in s:
            return sum(TextLength.approximate_char_width_em(c) for c in s)

        parts = re.split(r'(\$\$.*?\$\$)|(\$.*?\$)', s)
        total = 0.0
        for part in parts:
            if part is None:
                continue
            if part.startswith('$$') and part.endswith('$$'):
                total += TextLength.estimate_math_width(part[2:-2])
            elif part.startswith('$') and part.endswith('$'):
                total += TextLength.estimate_math_width(part[1:-1])
            else:
                visible = strip_html_tags(part)
                decoded = decode_entities(visible)
                total += sum(TextLength.approximate_char_width_em(c) for c in decoded)
        return total

    @staticmethod
    def estimate_math_width(tex: str) -> float:
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]|.', tex)
        width = 0.0
        for tok in tokens:
            if tok.startswith('\\'):
                name = tok[1:]
                if name in TextLength.TEX_SIMPLE_GLYPHS:
                    width += 0.6
                elif name in TextLength.TEX_WIDE:
                    width += 1.5
                elif name in TextLength.TEX_SPACING:
                    width += 0.4
                else:
                    width += 1.0  # unknown control sequences
            elif tok in '{}':
                continue  # grouping only
            elif tok in '^_':
                width += 0.3  # assume sub/superscript takes some space
            else:
                width += TextLength.approximate_char_width_em(tok)
        return width

    # original
    # @staticmethod
    # def text_display_len(s: str) -> int:
    #     """Estimate text display length in em of a string allowing for TeX constructs."""
    #     # note you DO WANT SPACES! So, no strip applied ever.
    #     if s.find('$') < 0:
    #         return len(s)
    #     parts = re.split(r'(\$\$.*?\$\$)|(\$.*?\$)', s)
    #     total = 0
    #     for part in parts:
    #         if part is None:
    #             continue
    #         if part.startswith('$$') and part.endswith('$$'):
    #             total += TextLength._estimate_math_width(part[2:-2])
    #         elif part.startswith('$') and part.endswith('$'):
    #             total += TextLength._estimate_math_width(part[1:-1])
    #         else:
    #             total += len(part)
    #     return total

    # @staticmethod
    # def _estimate_math_width(tex: str) -> int:
    #     tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]|.', tex)
    #     width = 0
    #     for tok in tokens:
    #         if tok.startswith('\\'):
    #             name = tok[1:]
    #             if name in TextLength.TEX_SIMPLE_GLYPHS:
    #                 width += 1
    #             elif name in TextLength.TEX_WIDE:
    #                 width += 3
    #             elif name in TextLength.TEX_SPACING:
    #                 width += 1
    #             else:
    #                 width += 2  # unknown control sequences
    #         elif tok in '{}^_':
    #             continue  # grouping, sub/superscripts: ignore
    #         else:
    #             width += 1
    #     return width


class Sparsify:
    """Sparsify multiindex rows and columns."""
    @staticmethod
    def sparsify(df, cs):
        out = df.copy()
        for i, c in enumerate(cs):
            mask = df[cs[:i + 1]].ne(df[cs[:i + 1]].shift()).any(axis=1)
            out.loc[~mask, c] = ''
        return out

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


class Indexing:
    """Changed column and level from a multi-index."""
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


class Width:
    """Adjust column widths based on heading widths."""

    @staticmethod
    def header_adjustment(df, min_widths, space, max_extra):
        """
        Fine-adjust heading for optimal config.spacing.

        Return a dict with per-column recommended width adjustments to avoid
        intra-word breaks and reduce overall header height.

        Parameters:
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
            # config.debug = pd.Series([col, min_acceptable[col], base_width, has_intra_word_break(col, base_width), extra0, max_extra,
            #     wrap(col,  base_width), extra],
            #     index=['col name', 'min acceptable', 'base_width (from data)', 'intra word break', 'extra0', 'max_extra', 'split', 'selected extra']).to_frame('Value')
            # display(config.debug)
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
            sol = Width.optimal_heading(input_df, space)
            adjustments.update(sol[1])
            logger.info('best solution: %s', sol)
        # global temp
        # temp = input_df
        return adjustments, input_df

    @staticmethod
    def optimal_heading(input_df: pd.DataFrame, total_es_budget: int) -> tuple[int, dict[str, int]]:
        """
        Optimize extra config.spacing for best heading.

        Finds the best way to allocate extra space to minimize max_lines in heading.

        Gemini solution.

        Args:
            input_df: DataFrame with 'col', 'extra', 'num_lines'.
            total_es_budget: The total extra space to allocate.

        Returns:
            A tuple: (min_max_lines, optimal_extra_allocation_per_column).

        Why this approach is effective:
        ---------------------------------

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


class TextOutput:
    """Convert dataframe to text, replaces pd.DataFrame.to_markdown."""

    @staticmethod
    def make_text_table(
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


class RichOutput:
    """Render to a rich table."""

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
