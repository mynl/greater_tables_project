"""
Find and process blobs of TeX.

Change target directory to find other blobs.
"""

from pathlib import Path
import re
import subprocess

import pandas as pd


class TeXMacros():
    """
    A class for dealing with TeX macros.

    made out of PublisherBase in blog_tools.py
    from great2.blog
    """

    _macros = r"""
\def\AA{\mathcal{A}}
\def\atan{\mathrm{atan}}
\def\AVaR{\mathsf{AVaR}}
\def\bbeta{\mathbf{\beta}}
\def\bb{\mathbf b}
\def\bm{\mathbf }
\def\biTVaR{\mathsf{biTVaR}}
\def\corr{\mathsf{Corr}}
\def\cov{\mathsf{cov}}
\def\cp{\mathsf{CP}}
\def\CTE{\mathsf{CTE}}
\def\CVaR{\mathsf{CVaR}}
\def\dint{\displaystyle\int}
\def\dsum{\displaystyle\sum}
\def\ecirc{\accentset{\circ} e}
\def\ecirc{\accentset{\circ} e}
\def\EPD{\mathsf{EPD}}
\def\ES{\mathsf{ES}}
\def\E{\mathsf{E}}
\def\FFF{\mathscr{F}}
\def\FF{\mathcal{F}}
\def\HH{\mathbf{H}}
\def\kpx{{{}_kp_x}}
\def\MM{\mathcal{M}}
\def\NN{\mathbb{N}}
\def\nudge{2}
\def\norm{}
\def\OO{\mathscr{O}}
\def\PPP{\mathscr{P}}
\def\PP{\mathsf{P}}
\def\Pr{\mathsf{Pr}}
\def\QQ{\mathsf{Q}}
\def\RR{\mathbb{R}}
\def\SD{\mathsf{SD}}
\def\TCE{\mathsf{TCE}}
\def\TVaR{\mathsf{TVaR}}
\def\Var{\mathsf{Var}}
\def\var{\mathsf{var}}
\def\VaR{\mathsf{VaR}}
\def\WCE{\mathsf{WCE}}
\def\ww{\mathbf{w}}
\def\XXX{\mathcal{X}}
\def\xx{\mathbf{x}}
\def\XX{\mathbf{X}}
\def\yy{\mathbf{y}}
\def\ZZZ{\mathcal{Z}}
\def\ZZ{\mathbb{Z}}
"""

    @staticmethod
    def process_tex_macros(text):
        """Expand standard general.tex macros in the text."""
        m, regex = TeXMacros.tex_to_dict(TeXMacros._macros.strip())
        return re.sub(regex, lambda x: m.get(x[0]), text, flags=re.MULTILINE)

    @staticmethod
    def tex_to_dict(text):
        """
        Convert text, a series of def{} macros into a dictionary
        returns the dictionary and the regex of all keys
        """
        smacros = text.split('\n')
        smacros = [TeXMacros.tex_splitter(i) for i in smacros]
        m = {i: j for (i, j) in smacros}
        regex = '|'.join([re.escape(k) for k in m.keys()])
        return m, regex

    @staticmethod
    def tex_splitter(x):
        """
        x is a single def style tex macro
        """
        x = x.replace('\\def', '')
        i = x.find('{')
        return x[:i], x[i + 1:-1]

def find_tex_snippeets(in_dir='\\S\\TELOS\\PIR\\docs',
                       out_file='tex_list.csv'):
    """Ripgrep / TeX macro expand list of TeX snippets."""
    result = subprocess.run(
        ['rg', '-N', '-o', '--no-filename', '-g', '*.md', r'\$.+?\$', in_dir],
        capture_output=True,
        text=True,
        check=True
    )
    output_text = result.stdout
    tm = TeXMacros()
    txt = tm.process_tex_macros(output_text)
    tex = txt.split('\n')
    stex = set(tex)
    stext = [i for i in stex if len(i) and i.find('\\PP') < 0 and i.find('$$') < 0]
    df = pd.DataFrame({'expr': stext})
    if out_file != '':
        p = Path(__file__).parent / out_file
        print(p)
        df.to_csv(p, encoding='utf-8')
    return df
