.. image:: https://img.shields.io/readthedocs/greater_tables_project
   :alt: Read the Docs

Release Notes
===============

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

TODO
=====

* Index aligners


https://shields.io/badges/read-the-docs