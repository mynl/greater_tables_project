"""
Create and display SVG files from TikZ pictures embedded in LaTeX.

Good for testing. Outputs are cached by hash. PDF→SVG uses pdf2svg.

GPT re-write of my old great2.blog code.
"""

import re
from pathlib import Path
from subprocess import run, Popen, PIPE
from IPython.display import SVG, display

from .gthasher import txt_short_hash


class Etcher:
    """Create PDF and SVG files from Tikz blocks."""
    # Full TeX preamble to generate a .fmt if needed
    _tex_template_full = r"""\documentclass[10pt, border=5mm]{standalone}
\usepackage{newtxtext,newtxmath}  % gpt recommended like STIX
%\usepackage{mathptmx}             % gpt like times roman
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{mathrsfs}
\usepackage{url}
\usepackage{tikz}
\usepackage{color}
\usetikzlibrary{arrows,calc,positioning,shadows.blur,decorations.pathreplacing}
\usetikzlibrary{automata,fit,snakes,intersections}
\usetikzlibrary{decorations.markings,decorations.text,decorations.pathmorphing,decorations.shapes}
\usetikzlibrary{decorations.fractals,decorations.footprints}
\usetikzlibrary{graphs,matrix,shapes.geometric}
\usetikzlibrary{mindmap,shadows,backgrounds,cd}
\dump
"""

    # Minimal template to embed user tikz
    _tex_template = r"""
\newcommand{{\I}}{{\vphantom{{lp}}}}   % fka grtspacer
\def\dfrac{{\displaystyle\frac}}
\def\dint{{\displaystyle\int}}
\begin{{document}}
{tikz_begin}{tikz_code}{tikz_end}
\end{{document}}
"""


    def __init__(self, txt, file_name='', base_path='.', tex_engine='pdflatex', debug=False):
        """Create object from txt, a TeX blob containing a tikzpicture."""
        self.txt = txt
        self.tex_engine = tex_engine
        self.base_path = Path(base_path).resolve()
        self.out_path = self.base_path / 'tikz'
        self.out_path.mkdir(exist_ok=True)
        file_name = file_name or  txt_short_hash(txt)
        self.file_path = self.out_path / file_name
        self.format_file = self.out_path / 'tikz_format.fmt'
        self.debug = debug

    def split_tikz(self):
        """Split text to extract the TikZ picture."""
        return re.split(r'(\\begin{tikz(?:cd|picture)}|\\end{tikz(?:cd|picture)})', self.txt)

    def ensure_format_file(self):
        """Create format file for faster compilation if missing."""
        if self.format_file.exists():
            return
        print('Etcher: building TeX format fmt file...', end ='')
        tmp = self.out_path / 'tikz_format.tex'
        tmp.write_text(self._tex_template_full, encoding='utf-8')
        cmd = [
            'pdflatex',
            f'-ini',
            f'-jobname={self.format_file.stem}',
            '&pdflatex',
            tmp.name,
            ]
        print(f'Running {" ".join(cmd)} to build format file...')
        (self.file_path.parent / 'make_format.bat').write_text(" ".join(cmd), encoding='utf-8')
        self.run_command(cmd, raise_on_error=True, cwd=self.out_path)
        # tidy up ... to some extent
        # tmp.unlink()
        (self.out_path / f'{self.format_file.stem}.log').unlink()
        print('...success...format file built', self.format_file.resolve())

    def process_tikz(self):
        """Compile TikZ to PDF and convert to SVG."""
        tikz_begin, tikz_code, tikz_end = self.split_tikz()[1:4]
        tex_code = self._tex_template.format(
            tikz_begin=tikz_begin,
            tikz_code=tikz_code,
            tikz_end=tikz_end
        )

        tex_path = self.file_path.with_suffix('.tex')
        tex_path.write_text(tex_code, encoding='utf-8')
        pdf_path = tex_path.with_suffix('.pdf')
        svg_path = tex_path.with_suffix('.svg')

        self.ensure_format_file()

        tex_cmd = [
            'pdflatex',
            "-interaction=nonstopmode",
            f'--fmt={self.format_file.stem}',
            f'--output-directory={str(tex_path.parent)}',
            str(tex_path)
        ]
        (tex_path.parent / 'make_tikz.bat').write_text(" ".join(tex_cmd), encoding='utf-8')
        if self.debug:
            print("Running:", " ".join(tex_cmd))
        if self.run_command(tex_cmd):
            raise ValueError('TeX failed to compile, not pdf or svg output.')
            # no tidying up
        else:
            # no error: continue
            svg_cmd = [
                # 'C:\\temp\\pdf2svg-windows\\dist-64bits\\pdf2svg',
                'pdf2svg',
                str(pdf_path),
                str(svg_path)
            ]
            if self.debug:
                print("Running:", " ".join(svg_cmd))
            self.run_command(svg_cmd, raise_on_error=True)

            if not self.debug:
                for ext in ('.tex', '.aux', '.log', '.pdf'):
                    path = tex_path.with_suffix(ext)
                    if path.exists():
                        path.unlink()

    def display(self):
        """Display the SVG in Jupyter."""
        display(SVG(self.file_path.with_suffix('.svg')))

    def run_command(self, command, raise_on_error=True, cwd=None):
        """Run command with subprocess and show output."""
        with Popen(command, cwd=cwd, stdout=PIPE, stderr=PIPE, universal_newlines=True) as p:
            stdout, stderr = p.communicate()
            if stdout and self.debug:
                print('Run command output ends\n', stdout.strip()[-250:])
            if stdout:
                if stdout.find('no output PDF file produced') > 0:
                    print("ERROR no pdf output\n"*5)
                    return -1
            if stderr:
                if raise_on_error:
                    raise RuntimeError(stderr.strip())
                else:
                    print(stderr.strip())
                    return -2
        return 0
