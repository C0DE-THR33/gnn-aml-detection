"""
Canonical typology definitions for the HI-Small AML dataset.

Per CONVENTIONS.md: never hardcode a typology name as a raw string in more
than one place. Import TYPOLOGIES / Typology from here instead.
"""

from enum import Enum


class Typology(str, Enum):
    """The 8 laundering typologies present in HI-Small_Patterns.txt."""

    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    CYCLE = "cycle"
    GATHER_SCATTER = "gather_scatter"
    SCATTER_GATHER = "scatter_gather"
    STACK = "stack"
    BIPARTITE = "bipartite"
    RANDOM = "random"

    # Residual bucket from Patterns.txt that doesn't map to a clean typology.
    # Excluded from per-typology fidelity scoring (see SRS FR-11).
    UNCLASSIFIED = "unclassified"


# Maps the raw strings as they appear in HI-Small_Patterns.txt to our enum.
# Update this mapping (not the enum itself) if the raw label spelling differs
# from what's assumed here once you inspect the real file.
RAW_LABEL_TO_TYPOLOGY = {
    "FAN-OUT": Typology.FAN_OUT,
    "FAN-IN": Typology.FAN_IN,
    "CYCLE": Typology.CYCLE,
    "GATHER-SCATTER": Typology.GATHER_SCATTER,
    "SCATTER-GATHER": Typology.SCATTER_GATHER,
    "STACK": Typology.STACK,
    "BIPARTITE": Typology.BIPARTITE,
    "RANDOM": Typology.RANDOM,
}

# The 8 typologies that get per-typology evaluation. UNCLASSIFIED is
# deliberately excluded — it is tracked but never scored (SRS FR-11 / 6.1 note).
EVALUATED_TYPOLOGIES = tuple(t for t in Typology if t != Typology.UNCLASSIFIED)


def normalize_typology(raw_label: str) -> Typology:
    """Map a raw pattern-file label to a Typology, defaulting to UNCLASSIFIED.

    Args:
        raw_label: The pattern type string as read from Patterns.txt.

    Returns:
        The corresponding Typology enum member, or Typology.UNCLASSIFIED if
        the raw label isn't recognized.
    """
    return RAW_LABEL_TO_TYPOLOGY.get(raw_label.strip().upper(), Typology.UNCLASSIFIED)
