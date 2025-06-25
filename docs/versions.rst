Versions
==========

5.0.0
-------
* Introduced ``knowledge_df`` as source of all formatting instructions
* ``df_tex`` and ``df_html`` computed before ``knowledge`` applying TeX and HTML specific transformations which are reflected in the estimated widths
* Introduced gtutilities to pull out text width estimation, cleaning and escaping, etc.
* Delete rich table output format?
* Moved logging setup to separate file, called from cli but optional for use in other situation. GPT recommended approach.

* test cases
    * tex and or html in data, index, columns, escaped/unescaped
    * long cols
    * text with multiindex columns

* TODO
    * self.padl and r / 12 in make html width adj s/b elsewhere

4.0.0
-------
* Moved constants out of magic strings and into config
* Renamed: gtetecher and `Etcher` class, gtconfig and `Conigurator`, gtfabrications and `Fabricator`
* Changed argument names in `Fabricator` to more align with dataframe and data terminology:
* Structuring docs

3.3.0
-------
* Added `tikz_` series of options to config: column and row separation,
  container_env (for e.g., sidewaystable), hrule and vrule indices.

3.2.0
-------
* Added more tex snippets!
* Refactored tikz and column width behavior

3.1.0
-------
* adjustments for auto format
* rearranged gtcore order of methods

3.0.0
-------

* config files / pydantic config input
* unified col width and info dataframe
* de-texing
* cli for config and writeout a csv etc.

* testdf suite
* Automated TeX to SVG

2.0.0
------

* **v2.0.0** solid release old-style, all-argument GT
* Better column widths
* Custom text output
* Rich table output

1.1.1
-------
* Added logo, updated docs.

1.1.0
------

* added ``formatters`` argument to pass in column specific formatters by name as a number (``n`` converts to ``{x:.nf}``, format string, or function
* Added ```tabs`` argument to provide column widths
* Added ``equal`` argument to provide hint that column widths should all be equal
* Added ``caption_align='center'`` argument to set the caption alignment
* Added ``large_ok=False`` argument, if ``False`` providing a dataframe with more than 100 rows throws an error. This function is expensive and is designed for small frames.

1.0.0
------

* Allow input via list of lists, or markdown table
* Specify overall float format for whole table
* Specify column alingment with 'llrc' style string
* ``show_index`` option
* Added more tests
* Docs updated
* Set tabs for width; use of width in HTML format.

0.6.0
------

* Initial release

Early development
-------------------

* 0.1.0 - 0.5.0: Early development
* tikz code from great.pres_manager
