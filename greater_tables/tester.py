from functools import partial
from pathlib import Path 
from subprocess import Popen, PIPE
import re

from greater_tables.core import GT as qd
from greater_tables.core_old import GT as qd_old
from greater_tables import Fabricator
# import pandas as pd


class GTTester:
    _preamble = r"""
% needs lualatex - uncomment for Wiley fonts
%\usepackage{fontspec}
%\setmainfont{Stix Two Text}
%\usepackage{unicode-math}
%\setmathfont{Stix Two Math}

\usepackage[
  a4paper, landscape,
  left=30mm,
  right=20mm,
  top=25mm,
  bottom=25mm,
  headheight=14pt,
  headsep=8mm,
  footskip=10mm
]{geometry}

\usepackage{url}
\usepackage{tikz}
\usepackage{color}
\usepackage{mathrsfs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{rotating}

\usetikzlibrary{arrows,calc,positioning,shadows.blur,decorations.pathreplacing}
\usetikzlibrary{automata}
\usetikzlibrary{fit}
\usetikzlibrary{snakes}
\usetikzlibrary{intersections}
\usetikzlibrary{decorations.markings,decorations.text,decorations.pathmorphing,decorations.shapes}
\usetikzlibrary{decorations.fractals,decorations.footprints}
\usetikzlibrary{graphs}
\usetikzlibrary{matrix}
\usetikzlibrary{shapes.geometric}
\usetikzlibrary{mindmap, shadows}
\usetikzlibrary{backgrounds}
\usetikzlibrary{cd}

%\def\dfrac{\displaystyle\frac}
%\def\dint{\displaystyle\int}
\def\I{\vphantom{Ag}}

\begin{document}
"""

    postamble = r"""\end{document}
"""

    def __init__(self, font_size='12pt', paper_size='a4paper', landscape=False, sep='\\clearpage',
                pyarrow=False):
        """
        Create multiple tikz tables for testing.

        Typical usage:

            import test
            tester = test.GTTester()
            tester.make_fabricator(rows=10, random=True)
            # or
            tester.make_fabricator(rows=10, random=False,
                    data_spec='ss3idxx',)  # other fab options
            tester.make_tex(n_test=4, file_name='testone.tex',
                    tex_engine='tectonic', execute=False)

        OR if you have your own blobs
            import test
            tester = test.GTTester()
            tester.append(blob1)
            ...
            tester.append(blobn)
            tester.make_tex(n_test=4, file_name='testone.tex',
                    tex_engine='tectonic', execute=False)


        """
        self.blobs = []
        # sort out header
        pa = GTTester._preamble
        if paper_size != 'a4paper':
            pa = pa.replace('a4paper', paper_size)
        if not landscape:
            pa = pa.replace(' landscape,', '')

        self.preamble = f'\\documentclass[{font_size}, {paper_size}]{{article}}\n' + pa
        self.sep = sep
        self.fab = Fabricator(pyarrow=pyarrow)
        self.maker = None
        # print('Now call make_fabricator to setup.')

    def append(self, blob):
        """Append an individual blob."""
        self.blobs.append(blob)

    def write_tex(self, file_name):
        """Write out combined tex file."""
        path = Path(file_name)
        if not path.suffix == '.tex':
            path = path.with_suffix('.tex')

        if path.exists():
            path.unlink()

        bits = [self.preamble]
        divider = f'\n{self.sep}\n' + "% " + "=" * 50 + '\n'
        txt = divider.join(self.blobs)

        bits.append(txt)
        bits.append(self.postamble)

        txt = '\n'.join(bits)
        # tidy up a bit
        txt = re.sub(r'\n\n+', '\n\n', txt)
        path.write_text(txt, encoding='utf-8')
        print(f'File {path} written, {len(self.blobs)} blobs.')
        return path

    def make_fabricator(self, rows=10, random=False,
                    data_spec='', omit='h', **kwargs):
        """
        Make function that creates the dataframes. This
        can then be called with different greater_table options.

        kwargs passed to fab.make if NOT random.

            data_spec,
            index_levels=1,
            index_names=None,
            column_groups=1,
            column_levels=1,
            column_names=None,
            metric_name_spec='',
            decorate=False,
            simplify=True,
            oversample=1,

        metric_name_spec = '' or tuple.list of names, or a spec

        Data types

            d   date
            f   float
            h   hash
            i   integer
            l   log float (greater range than float)
            m   year - month
            p   path (filename)
            r   ratio (smaller floats, for percents)
            sx  string length x, s for one word
            t   time
            v   very large range float
            x   tex text - an equation
            y   year
        """
        if random:
            self.maker = partial(self.fab.random, rows=rows, omit=omit)
        else:
            assert data_spec != "", "Must pass data_spec in non-random mode."
            # hash is a pain in the ass
            self.maker = partial(self.fab.make, rows, data_spec, **kwargs)

    def make_tex(self, n_test, file_name, tex_engine='tectonic',
            execute=False, **kwargs):
        """
        Create the tables and assemble into tex file.

        kwargs are passed to GT:

            'caption_align': 'center',
            'cast_to_floats': True,
            'debug': False,

            'default_date_str': '%Y-%m-%d',
            'default_float_str': '{x:,.3f}',
            'default_formatter': None,
            'default_integer_str': '{x:,d}',
            'default_ratio_str': '{x:.1%}',

            'equal': False,

            'font_body': 0.9,
            'font_bold_index': False,
            'font_caption': 1.1,
            'font_head': 1.0,

            'header_alignment': 'few',
            'header_row': True,

            'hrule_widths': (1.0, 1.0, 1.0),

            'large_ok': False,
            'large_warning': 50,

            'max_str_length': -1,
            'max_table_inch_width': 8.0,

            'padding_trbl': None,
            'pef_lower': -3,
            'pef_precision': 3,
            'pef_upper': 6,

            'spacing': 'medium',

            'sparsify': True,
            'sparsify_columns': True,

            'table_float_format': None,
            'table_font_pt_size': 11.0,
            'table_hrule_width': 1,
            'table_vrule_width': 1,
            'table_width_header_adjust': 0.1,
            'table_width_header_relax': 10.0,
            'table_width_mode': 'explicit',

            'tex_to_html': None,
            'tikz': True,
            'tikz_column_sep': 1.0,
            'tikz_container_env': 'table',
            'tikz_escape_tex': True,
            'tikz_hrule': None,
            'tikz_latex': None,
            'tikz_post_process': '',
            'tikz_row_sep': 0.25,
            'tikz_scale': 1.0,
            'tikz_vrule': None,
            'vrule_widths': (1.0, 1.0, 1.0)
        """
        for i in range(n_test):
            df = self.maker()
            q = qd(df)
            txt = q.make_tikz()
            self.append(txt)
        print(f'{n_test} dataframes created.')

        tex_path = self.write_tex(file_name)
        print(f'Saved to {tex_path.name}.')
        if execute:
            self.process(tex_path)
            print(f'Created pdf file {tex_path}')

    def process(self, tex_path, tex_engine='tectonic'):
        """Run TeX."""
        tex_path = Path(tex_path).resolve()
        if tex_engine == 'pdflatex':
            # faster with template
            command = ['pdflatex',
                       f'--output-directory={str(tex_path.parent.resolve())}',
                       str(tex_path.resolve())]
        elif tex_engine == 'lualatex':
            # for STIX fonts, no template
            command = ['lualatex',
                       f'--output-directory={str(tex_path.parent.resolve())}',
                       str(tex_path.resolve())]
        elif tex_engine == 'tectonic':
            command = ['tectonic',
                        "-X",
                        "compile",
                        str(tex_path),
                        '--keep-logs',
                        '--keep-intermediates',
                        '-p']
        else:
            raise ValueError('Unknown tex_engine.')
        GTTester.run_command(command, flag=False)
        # to recreate
        (tex_path.parent /
            'make_tikz.bat').write_text(" ".join(command))

    @staticmethod
    def gt_config():
        """Dump gt config."""
        return qd(""" """).config.model_dump()

    # from blog tools ====================================================================
    @staticmethod
    def run_command(command, flag=True):
        """
        Run a command and show results. Allows for weird xx behavior

        :param command:
        :param flag:
        :return:
        """
        with Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True) as p:
            line1 = p.stdout.read()
            line2 = p.stderr.read()
            exit_code = p.poll()
            if line1:
                print('\n' + line1[-250:])
            if line2:
                if flag:
                    raise ValueError(line2)
                else:
                    print(line2)
        return exit_code
