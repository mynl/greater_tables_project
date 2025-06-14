"""
Configuration model and utilities for GreaterTables.

Defines the `GTConfigModel` schema using Pydantic, which acts as the single
source of truth for default values, validation, and structure of all table-rendering options.

Also includes functions for writing editable config templates and loading from YAML.
"""


from pathlib import Path
from typing import Optional, Union, Literal
import yaml

from pydantic import BaseModel, Field, ValidationError, ConfigDict
import yaml


class GTConfigModel(BaseModel):
    """
    Configuration model for GreaterTables.

    This class defines all configurable options for controlling the formatting
    and rendering of tables in HTML, text, and LaTeX outputs.

    Each field has a default value and is validated using Pydantic. You can load
    configuration from a YAML file or create it programmatically. Use this model
    as the authoritative source of valid configuration fields.

    :Usage:

        >>> from greater_tables.gtconfig import GTConfigModel
        >>> cfg = GTConfigModel(font_size="1.2em", caption_align="left")

    :see also: ``GTConfig`` for loading from YAML with overrides.
              ``gt write-template`` CLI command to generate a default config file.
    """
    # immutable
    model_config = ConfigDict(frozen=True)
    default_integer_str: str = Field(
        "{x:,d}", description="Format f-string for integers. Example: '{x:,d}'"
    )
    default_float_str: str = Field(
        "{x:,.3f}", description="Format f-string for floats. Example: '{x:,.3f}'"
    )
    default_date_str: str = Field(
        "%Y-%m-%d", description="Format string for dates (no braces or 'x'). Example: '%Y-%m-%d'"
    )
    default_ratio_str: str = Field(
        "{x:.1%}", description="Format f-string for ratios. Example: '{x:.1%}'"
    )
    default_formatter: Optional[str] = Field(
        None, description="Optional fallback formatter f-string"
    )

    table_float_format: Optional[str] = Field(
        None, description="Float format string for the entire table; overrides column-specific formats"
    )
    table_hrule_width: int = Field(
        1, description="Width of top, bottom, and header horizontal rules"
    )
    table_vrule_width: int = Field(
        1, description="Width of vertical rule separating index from body"
    )
    hrule_widths: Optional[tuple[float, float, float]] = Field(
        (0, 0, 0), description="Tuple of three floats for horizontal rule widths (for multiindex use)"
    )
    vrule_widths: Optional[tuple[float, float, float]] = Field(
        (0, 0, 0), description="Tuple of three floats for vertical rule widths (for multiindex columns)"
    )

    sparsify: bool = Field(
        True, description="If True, sparsify index columns (recommended)"
    )
    sparsify_columns: bool = Field(
        True, description="If True, sparsify column headers using colspans"
    )

    spacing: str = Field(
        "medium", description="Shorthand for cell padding. One of: 'tight', 'medium', 'wide'"
    )
    padding_trbl: Optional[tuple[int, int, int, int]] = Field(
        None, description="Manual padding in the order (top, right, bottom, left)"
    )

    tikz_scale: float = Field(
        1.0, description="Scaling factor applied to LaTeX TikZ tables"
    )
    font_body: float = Field(
        0.9, description="Font size for body text (in em units)"
    )
    font_head: float = Field(
        1.0, description="Font size for header text (in em units)"
    )
    font_caption: float = Field(
        1.1, description="Font size for caption text (in em units)"
    )
    font_bold_index: bool = Field(
        False, description="If True, make index columns bold"
    )

    pef_precision: int = Field(
        3, description="Precision for engineering format (digits after decimal)"
    )
    pef_lower: int = Field(
        -3, description="Lower threshold: apply engineering format if abs(x) < 10**pef_lower"
    )
    pef_upper: int = Field(
        6, description="Upper threshold: apply engineering format if abs(x) > 10**pef_upper"
    )

    cast_to_floats: bool = Field(
        True, description="If True, cast non-integer, non-date columns to float where possible"
    )
    header_row: bool = Field(
        True, description="If True, use the first row as header; False disables header row"
    )
    # tabs: Optional[Union[list[float], float, int]] = Field(
    #     None, description="Column widths in characters or ems; None triggers auto-calculation"
    # )
    equal: bool = Field(
        False, description="If True, force equal column widths (may be ignored if conflicting)"
    )

    caption_align: str = Field(
        "center", description="Alignment of the caption text"
    )
    large_ok: bool = Field(
        False, description="If True, allow full rendering of large tables without truncation"
    )

    max_str_length: int = Field(
        -1, description="Maximum length for stringified objects (e.g. nested DataFrames); -1 = unlimited"
    )

    max_table_width: int = Field(
        200, description="Maximum table width for markdown/text output mode"
    )
    table_width_mode: Literal["explicit", "natural", "breakable", "minimum"] = Field(
        "explicit",
        description=(
            "Mode for determining table width. "
            "'explicit': fixed width using max_table_width; "
            "'natural': each cell fits its full content; "
            "'breakable': wrap breakable strings; "
            "'minimum': also wraps dates or float-like cells"
        )
    )
    table_width_header_adjust: float = Field(
        0.1, description="Proportion of width allocated to headers to balance content width"
    )
    table_width_header_relax: float = Field(
        10.0, description="Extra characters allowed per column heading to help header wrapping"
    )
    debug: bool = Field(False, description="Run in debug mode with more reporting, include internal ID in caption and use colored output lines")

    def write_template(self, path: Path):
        """Generate a clean default config file at the given path."""
        path = Path(path)
        yaml_str = yaml.dump(self.model_dump(), sort_keys=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml_str, encoding="utf-8")


def write_template(path: Path):
    """Generate a clean default config file at the given path."""
    path = Path(path)
    cfg = GTConfigModel()
    yaml_str = yaml.dump(cfg.model_dump(), sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_str, encoding="utf-8")
