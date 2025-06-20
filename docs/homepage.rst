Greater Tables: Purpose and Process
=====================================

1. Introduction and Purpose
-------------------------------

.. raw analysis selected/curated display

**Greater Tables** is a Python-based tool for producing high-quality, static display tables—intended for use in journal articles, books, formal reports, and printed financial statements. Its focus is on clarity, precision, and typographic consistency in **black-and-white, presentation-grade tables**. If you need visual embellishments like colors, sparklines, or interactive filters, you're looking for a different kind of tool. Greater Tables is about making numbers clear—not making them move.

The goal is to bridge the final step between analysis and publication: turning structured data into a table that looks right on the page. That means getting the **spacing, alignment, wrapping, and formatting** right so the reader can focus on the content—not fight with the layout.

It’s important to emphasize that Greater Tables is **not** a charting library or an exploratory data analysis tool. Those serve different purposes. Charts help readers understand patterns and trends—often at the cost of precision. Tables are for **exact values**, **structured comparison**, and **decisions**.

Greater Tables is built for **static, print-ready output**. This is not a dashboarding tool, and it does not attempt to support interactivity. If you're preparing data for decision-makers, regulators, or publication, Greater Tables helps you present that data with clarity, consistency, and dignity. It turns structured data into final-form tables: faithful to the analysis, respectful of the reader.

.. admonition::  Charts vs. Tables

    Tables and charts serve different cognitive functions.

        * **Charts** are tools for exploration, discovery, and communication of patterns. They help identify trends and relationships, especially where precision is secondary.
        * **Tables** are tools for reference, validation, and decision-making. They are used where exact values matter and where the structure is already known.

    You might use a chart to explain seasonal trends to a client. You use a table to calculate bonuses, approve budgets, or submit results to a regulator.

    Greater Tables is for that second case: high-precision, high-trust, high-clarity tabular presentation.

2. Anatomy of a Table
---------------------------

A well-formed table has three structural components:

1. **The data**—a rectangular array of values, typically numeric or categorical.
2. **Row labels**—one or more leading columns that organize the data by grouping rows, often hierarchically (e.g., country → region → city).
3. **Column labels**—a corresponding structure that organizes the columns, such as year → quarter → month.

Additional elements such as a **title**, **caption**, or **footnotes** may accompany the table but are not considered part of the data itself. Greater Tables encourages separating descriptive or contextual information into captions, preserving the purity of the table structure.

While most software merges row labels into the data table and treats column labels as headers, Greater Tables gives both equal attention. Each can carry meaningful hierarchy. The distinction between rows and columns is often arbitrary—in many tables, either axis could be transposed without changing the semantics. Layout and print constraints usually determine which becomes which.

.. admonition::  From Raw Data to Presentation Table

    Before using Greater Tables, the dataset goes through several stages of preparation:

    1. **Raw data**—often includes many columns, internal naming, and structure optimized for computation.
    2. **Analysis dataset**—cleaned, subsetted, and possibly aggregated; new variables may be computed here.
    3. **Derived elements**—secondary metrics like change over time, percentages of subtotal, or indexed values. These don't reflect direct observations but are built from them.
    4. **Pre-presentation table**—a tidy, well-labeled subset of columns, in the desired order and naming scheme.
    5. **Presentation**—Greater Tables takes this prepped table and handles only layout and typography.

.. admonition::  Tidy Data

    Greater Tables assumes your table follows the tidy data model:

    * Each column is a variable.
    * Each row is an observation.
    * Each cell contains one value.

    Hierarchical data (e.g., region → city, or year → month) is expressed via multiple columns or MultiIndexes. This structure is always **flat in form**, even if hierarchical in meaning.

.. admonition::  Pandas Index vs. SQL Index

    In pandas, the *Index* is a labeling tool used to identify rows and columns for selection, alignment, and display. It is semantic. In SQL, an *index* is a performance structure used for speeding up lookups. Same word, very different purpose.

3. Labeling, Formatting, and Layout
------------------------------------

Once a tidy, pre-presentation table is prepared, Greater Tables applies formatting decisions:

* **Column widths** are calculated from both the data values and the label lengths.
* **Alignment** is conventional: text is left-aligned, numbers right-aligned, and dates centered.
* **Text wrapping** can be applied, ideally with ragged-right edges and minimal hyphenation.
* **Number formatting** supports thousands separators, consistent decimal places, and suppression of floating-point artifacts.
* **Monospaced fonts** are recommended for numbers to make differences in magnitude visually apparent.
* **Semantic formatting** is encouraged: margins as a percentage of sales, changes over time, or indexed values.

Greater Tables does **not** modify the structure of the data. It does not sort, filter, rename, or pivot. It simply takes the table and **renders it with typographic precision**, making good decisions about space, alignment, and clarity.

.. admonition:: Meta-Rows and Derived Columns

    In final tables, it’s common to introduce elements that are not raw observations:

    * **Subtotals** and **grand totals**: rows that summarize other rows.
    * **Percentage-of-total** or **difference-from-baseline**: columns derived from multiple observations.
    * **Headers or separators**: rows that serve as group titles.

    These are structural enhancements introduced during presentation, not part of the tidy data itself. Greater Tables renders them as-is, assuming you’ve added them deliberately to improve clarity.


.. admonition:: Time, State, and Change

    Time is a special axis.

    Most datasets report either:

    * **States** at a point in time (e.g., balance, inventory, weight), or
    * **Changes** over a period (e.g., revenue, flow, weight loss).

    These correspond to **point-in-time** vs. **over-time** measurements. Both can be stored in tidy format, but mixing them requires care. Their time labels mean different things.

    Other axes (e.g., elevation vs. change in elevation, treatment A vs. B, document version X vs. Y) follow similar logic, but time remains the most natural and universal case. Derived quantities like differences or growth rates reflect *relationships* across time, not independent observations. They must be labeled accordingly.

    The analogy to **balance sheet vs. income statement** is instructive:

    * A balance sheet gives a snapshot at a single date.
    * An income statement measures change between two dates.
    * Both are valid, but they require different structures—and cannot be naively combined.

4. Output Consistency Across Formats
----------------------------------------

Greater Tables produces tables in three formats:

* **Text** (for console or plain-text rendering),
* **HTML** (for web or rich email output),
* **TeX/PDF** (for inclusion in LaTeX documents).

Each output is designed to preserve the structure, alignment, and formatting choices defined by the table's metadata. The layout engine adapts to the output medium but **never alters the underlying table**. This ensures visual consistency across formats.

Each backend respects:

* Column widths and wrapping constraints,
* Font styles (monospaced for numbers),
* Multi-level headers and index structure,
* Alignment and spacing.

This guarantees that the same table, rendered in multiple formats, carries the same logic, appearance, and communicative power.
