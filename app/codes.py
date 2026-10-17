"""What the one-letter codes in data/mushrooms.csv mean.

The CSV stores every feature as a single letter ("p" for pungent, "f" for foul...). Training
translates the letters into these words, so the model, the API and its users all speak words.
"""

from __future__ import annotations

COLOURS = {
    "n": "brown",
    "b": "buff",
    "c": "cinnamon",
    "g": "gray",
    "r": "green",
    "o": "orange",
    "p": "pink",
    "u": "purple",
    "e": "red",
    "w": "white",
    "y": "yellow",
    "k": "black",
    "h": "chocolate",
}
SURFACES = {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "k": "silky"}


def colours(*letters: str) -> dict[str, str]:
    return {letter: COLOURS[letter] for letter in letters}


def surfaces(*letters: str) -> dict[str, str]:
    return {letter: SURFACES[letter] for letter in letters}


# Feature name (as the API spells it) -> {letter in the CSV: word}.
CODES: dict[str, dict[str, str]] = {
    "cap_shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken",
    },
    "cap_surface": surfaces("f", "g", "y", "s"),
    "cap_color": colours("n", "b", "c", "g", "r", "p", "u", "e", "w", "y"),
    "bruises": {"t": "yes", "f": "no"},
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy",
    },
    "gill_attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill_spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill_size": {"b": "broad", "n": "narrow"},
    "gill_color": colours("k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"),
    "stalk_shape": {"e": "enlarging", "t": "tapering"},
    "stalk_root": {
        "b": "bulbous",
        "c": "club",
        "u": "cup",
        "e": "equal",
        "z": "rhizomorphs",
        "r": "rooted",
        "?": "missing",  # 2,480 mushrooms have no recorded stalk root
    },
    "stalk_surface_above_ring": surfaces("f", "y", "k", "s"),
    "stalk_surface_below_ring": surfaces("f", "y", "k", "s"),
    "stalk_color_above_ring": colours("n", "b", "c", "g", "o", "p", "e", "w", "y"),
    "stalk_color_below_ring": colours("n", "b", "c", "g", "o", "p", "e", "w", "y"),
    "veil_color": colours("n", "o", "w", "y"),
    "ring_number": {"n": "none", "o": "one", "t": "two"},
    "ring_type": {
        "c": "cobwebby",
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
        "s": "sheathing",
        "z": "zone",
    },
    "spore_print_color": colours("k", "n", "b", "h", "r", "o", "u", "w", "y"),
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary",
    },
    "habitat": {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods",
    },
}

# veil-type is "partial" for every one of the 8,124 mushrooms: it cannot tell them apart.
DROPPED = ["veil_type"]
FEATURES = list(CODES)
