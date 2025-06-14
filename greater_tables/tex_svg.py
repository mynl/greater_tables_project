"""
Create and display svg files from tikz tex tables.

Good for testing.

From great2.blog
"""

from datetime import datetime
import pandas as pd
from pathlib import Path
import re
import yaml
from itertools import count
from subprocess import Popen, PIPE
from IPython.display import display, Markdown, SVG

from . hasher import txt_short_hash


class TikzProcessor():
    _tex_template_full = """\\documentclass[10pt, border=5mm]{{standalone}}

% needs lualatex - uncomment for Wiley fonts
%\\usepackage{{fontspec}}
%\\setmainfont{{Stix Two Text}}
%\\usepackage{{unicode-math}}
%\\setmathfont{{Stix Two Math}}

\\usepackage{{amsfonts}}
\\usepackage{{url}}
\\usepackage{{tikz}}
\\usepackage{{color}}
\\usetikzlibrary{{arrows,calc,positioning,shadows.blur,decorations.pathreplacing}}
\\usetikzlibrary{{automata}}
\\usetikzlibrary{{fit}}
\\usetikzlibrary{{snakes}}
\\usetikzlibrary{{intersections}}
\\usetikzlibrary{{decorations.markings,decorations.text,decorations.pathmorphing,decorations.shapes}}
\\usetikzlibrary{{decorations.fractals,decorations.footprints}}
\\usetikzlibrary{{graphs}}
\\usetikzlibrary{{matrix}}
\\usetikzlibrary{{shapes.geometric}}
\\usetikzlibrary{{mindmap, shadows}}
\\usetikzlibrary{{backgrounds}}
\\usetikzlibrary{{cd}}

% really common macros
\\newcommand{{\\grtspacer}}{{\\vphantom{{lp}}}}

\\def\\dfrac{{\\displaystyle\\frac}}
\\def\\dint{{\\displaystyle\\int}}

\\begin{{document}}

{tikz_begin}{tikz_code}{tikz_end}

\\end{{document}}
"""
    # --------------------------------------------
    _tex_template = """
% really common macros
\\newcommand{{\\grtspacer}}{{\\vphantom{{lp}}}}

\\def\\dfrac{{\\displaystyle\\frac}}
\\def\\dint{{\\displaystyle\\int}}

\\begin{{document}}

{tikz_begin}{tikz_code}{tikz_end}

\\end{{document}}
"""

    def split_tikz(self):
        """
        Split text to get the tikzpicture. Format is

        initial text pip then groups of four:

        1. begin tag ``(1::4)``
        2. tikz code ``(2::4)``
        3. end tag   ``(3::4)``
        4. non-related text ``(4::4)``

        """
        return re.split(r'(\\begin{tikz(?:cd|picture)}|\\end{tikz(?:cd|picture)})', self.txt)

    def __init__(self, txt, base_path='.', tex_engine='pdflatex'):
        """
        TikzProcessor (from TikzConvertyer): process a tex tikz text string into svg.
        The program

        * creates a pdf and svg from the tikz blob

        lualatex is more robust, but slower...
        pdflatex can't handle the fancy wiley fonts

        """
        self.txt = txt
        self.tex_engine = tex_engine
        # directory for TeX and images
        self.base_path = Path(base_path).resolve()
        self.out_path = self.base_path / 'tikz'
        self.out_path.mkdir(exist_ok=True)
        self.file_path = self.out_path / txt_short_hash(txt)

    def process_tikz(self, verbose=False):
        """
        Process the tikz into pdf and svg
        """
        # container contains a tikzpicture
        svg_path = self.file_path.with_suffix('.svg')
        tex_path = self.file_path.with_suffix('.tex')

        # make tex code for a stand-alone document
        tikz_begin, tikz_code, tikz_end = self.split_tikz()[
            1:4]
        tex_code = self._tex_template.format(
            tikz_begin=tikz_begin, tikz_code=tikz_code, tikz_end=tikz_end)
        tex_path.write_text(tex_code, encoding='utf-8')
        print(
            f'TIKZ: created temp file = {tex_path.name}')
        pdf_file = tex_path.with_suffix('.pdf')
        print(f'TIKZ: Update pdf file')
        if self.tex_engine == 'pdflatex':
            # faster with template
            # TODO EVID hard coded template
            template_path = Path('tikz_format.fmt')
            assert template_path.exists()
            template = str(template_path)
            command = ['pdflatex', f'--fmt={template}',
                       f'--output-directory={str(tex_path.parent.resolve())}',
                       str(tex_path.resolve())]
        else:
            # for STIX fonts, no template
            command = ['lualatex',
                       f'--output-directory={str(tex_path.parent.resolve())}',
                       str(tex_path.resolve())]
        if verbose:
            print(f'TIKZ: TeX Command={" ".join(command)}')
        TikzProcessor.run_command(command)
        # to recreate
        (tex_path.parent /
         f'make_tikz.bat').write_text(" ".join(command))
        if verbose:
            print(
                f'TIKZ: Creating svg file for Tikz (using new pdf2svg util)')
        # https://github.com/jalios/pdf2svg-windows
        command = [
            'C:\\temp\\pdf2svg-windows\\dist-64bits\\pdf2svg',
            str(pdf_file.resolve()), str(svg_path.resolve())]
        # seems to return info on stderr?
        if verbose:
            print(f'PDF->SVG: {" ".join(command)}')
        TikzProcessor.run_command(command, flag=False)
        if not verbose:
            # tidy up
            tex_path.unlink()
            tex_path.with_suffix('.aux').unlink()
            tex_path.with_suffix('.log').unlink()
            pdf_file.unlink()

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

    def display(self):
        """display in Jupyter Lab."""
        display(SVG(self.file_path.with_suffix('.svg')))
