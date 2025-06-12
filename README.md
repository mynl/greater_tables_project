sort out the variety of readmes... this is the main one

# v 3.0 update

* config files
* unified col width and info dataframe
* de-texing
* cli for config and writeout a csv etc.
* tests

# TODO

* Ratio cols with multi index columns
* % in tex output - never allow comments?
* center / left / right table output -> CSS
* ?Option to hide index
* Bring over the roll your own logger

***

# from GPT


Absolutely—here’s a structured summary of everything we’ve covered, organized by topic.

---

## 🧱 Project Structure & Philosophy

* Your `GreaterTables` class formats a `pandas.DataFrame` to **HTML, text, or LaTeX**.
* The class is **immutable**: formatting is fixed at construction time, like a pure value object.
* You avoid branchy, incremental APIs (like `ggplot`) and prefer creating fresh objects.
* You wanted a way to handle growing config complexity—→ led to a **YAML config + Pydantic schema** design.

---

## 📁 Project Layout

```
greater_tables_project/
├── greater_tables/
│   ├── __init__.py
│   ├── gtconfig.py          ← config model + loader
│   ├── gtcore.py            ← GreaterTables class
│   └── defaults/
│       └── config_template.yaml
├── tests/
├── pyproject.toml
```

* `GTConfigModel` = schema + default source of truth
* `GTConfig` = singleton loader and validator
* `config_template.yaml` = editable fallback + documentation base

---

## 🔧 Config Management

* All defaults and types are declared in `GTConfigModel` (Pydantic).
* Config is **loaded from YAML**, validated by `GTConfigModel`.
* You can **generate** a valid config file from the model using `.model_dump() → YAML`.
* Singleton pattern (`GTConfig.__new__`) caches the config at runtime.

### Helpers

* `GTConfig().get(overrides=...)` gives a safe, override-able config
* `write_template(path)` writes a default config YAML for user to edit

---

## 🛠 Git Workflow (Solo Dev, Linear)

* Use **tags** (`git tag v0.2.0`) to label stable versions
* Use **`git reset --hard <tag>`** to roll back and discard later commits
* Avoid branches entirely—keep a **single linear history**
* Tags let you bounce around safely, with names instead of hashes
* Releases on GitHub are tags + metadata, optional for publishing

---

## ⚙️ CLI Tool

* Built with `click`, with subcommands:

  * `gt render data.csv --format html`
  * `gt write-template`
* Reads any Pandas-supported file (`.csv`, `.feather`, `.pkl`, etc.)
* Outputs to console or to file
* Uses current config by default, or override with `--config path.yaml`

---

## 🧠 Design Principles You’re Following

| Principle                    | Your Approach                               |
| ---------------------------- | ------------------------------------------- |
| Immutability                 | `GT(df, config)` is fixed once created      |
| Separation of concerns       | `GTConfigModel` holds defaults/types        |
| Config as code/documentation | `config_template.yaml` generated from model |
| CLI-first mindset            | `click` used to expose functionality        |
| Linear Git workflow          | Tags for rollback, no branches              |

---

Let me know if you want me to generate:

* a Markdown doc for contributors
* a `.bat` script to roll back to a tag
* test scaffolding or release automation

You're in great shape. Gum-level perfection achieved.



***

# OLD

# Greater Tables

Creating presentation quality tables from pandas dataframes is frustrating. It is hard to left-align text and right-align numbers using pandas `display` or `df.to_html`. The  `great_tables` package does a really nice job with pandas and polars dataframes but does not support indexes or TeX output. 

This package provides consistent HTML and TeX table output with flexible type-based formatting, and table rules. Neither output relies on the pandas `to_html` or `to_latex` functions. TeX output uses Tikz tables for very tight control over layout and grid lines. The package is designed for use in Jupyter Lab notebooks Quarto documents.

Usage: the main class `GT` should be subclassed to set appropriate defaults for your project. `sGT` provides an example.

The project is currently in **beta** status. HTML output is better developed than TeX.

## The Name

Obviously, the name is a play on the `great_tables` package. But, I have
been maintaining a set of macros called
[GREATools](https://www.mynl.com/old/GREAT/home.html) (generalized,
reusable, extensible actuarial tools) in VBA and Python since the late
1990s, and call all my macro packages "GREAT".

## Installation

``` python
pip install greater-tables
```

## Examples

The following example shows quite a hard table. It is formatted using
the `sGT` class, which is a subclass of `GT` with a few defaults set.

``` {.python .cell-code}
import pandas as pd
import numpy as np
from greater_tables import sGT
level_1 = ["Group A", "Group A", "Group B", "Group B", 'Group C']
level_2 = ['Sub 1', 'Sub 2', 'Sub 2', 'Sub 3', 'Sub 3']

multi_index = pd.MultiIndex.from_arrays([level_1, level_2])

start = pd.Timestamp.today().normalize()  # Today's date, normalized to midnight
end = pd.Timestamp(f"{start.year}-12-31")  # End of the year

hard = pd.DataFrame(
{'x': np.arange(2020, 2025, dtype=int), 
'a': np.array((100, 105, 2000, 2025, 100000), dtype=int),
'b': 10. ** np.linspace(-9, 9, 5),
'c': np.linspace(601, 4000, 5),
'd': pd.date_range(start=start, end=end, periods=5),
'e': 'once upon a time, risk is hard to define, not in Kansas anymore, neutrinos are hard to detect,  $\\int_\\infty^\\infty e^{-x^2/2}dx$ is a hard integral'.split(',')
}).set_index('x')
hard.columns = multi_index
sGT(hard, 'A hard table.')
```

![HTML output.](img/hard-html.png)

![TeX output.](img/hard-tex.png)

The output illustrates:

-   Quarto or Jupyter automatically the class's `_repr_html_` method (or
    `_repr_latex_` for pdf/TeX/Beamer output), providing seamless
    integration across different output formats.
-   Text is left-aligned, numbers are right-aligned.
-   The index is displayed, was detected as likely years, and formatted
    without a comma separator.
-   The first column of integers does have a comma thousands separator.
-   The second column of floats spans several orders of magnitude and is
    formatted using Engineering format, n for nano through G for giga.
-   The third column of floats is formatted with a comma separator and
    two decimals, based on the average absolute value.
-   The fourth column of date times is formatted as ISO standard dates
    (not date times).
-   The vertical lines separate the levels of the column multiindex. The
    subgroups are a little tricky.

More coming soon.

## Documentation

Available on
[readthedocs](https://greater-tables-project.readthedocs.io/en/latest).

## Versions

### 1.1.1
* Added logo, updated docs.

### 1.1.0
