"""What the API accepts and returns. FastAPI validates requests against these and documents
them at /docs."""

# No `from __future__ import annotations` here: the choice(...) types are built at import time.
from typing import Literal

from pydantic import BaseModel, Field

from app.codes import CODES


def choice(feature: str):
    """Only the words this feature can take (from app/codes.py); anything else is a 422."""
    return Literal[tuple(CODES[feature].values())]


# The first mushroom in the dataset: poisonous (pungent smell, black spore print).
EXAMPLE_MUSHROOM = {
    "cap_shape": "convex",
    "cap_surface": "smooth",
    "cap_color": "brown",
    "bruises": "yes",
    "odor": "pungent",
    "gill_attachment": "free",
    "gill_spacing": "close",
    "gill_size": "narrow",
    "gill_color": "black",
    "stalk_shape": "enlarging",
    "stalk_root": "equal",
    "stalk_surface_above_ring": "smooth",
    "stalk_surface_below_ring": "smooth",
    "stalk_color_above_ring": "white",
    "stalk_color_below_ring": "white",
    "veil_color": "white",
    "ring_number": "one",
    "ring_type": "pendant",
    "spore_print_color": "black",
    "population": "scattered",
    "habitat": "urban",
}


class Mushroom(BaseModel):
    """What you can see (and smell) on one mushroom: the 21 features the model was trained on."""

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [EXAMPLE_MUSHROOM]}}

    cap_shape: choice("cap_shape")
    cap_surface: choice("cap_surface")
    cap_color: choice("cap_color")
    bruises: choice("bruises") = Field(description="Does it bruise when handled?")
    odor: choice("odor")
    gill_attachment: choice("gill_attachment")
    gill_spacing: choice("gill_spacing")
    gill_size: choice("gill_size")
    gill_color: choice("gill_color")
    stalk_shape: choice("stalk_shape")
    stalk_root: choice("stalk_root") = Field(description="'missing' if it was not recorded")
    stalk_surface_above_ring: choice("stalk_surface_above_ring")
    stalk_surface_below_ring: choice("stalk_surface_below_ring")
    stalk_color_above_ring: choice("stalk_color_above_ring")
    stalk_color_below_ring: choice("stalk_color_below_ring")
    veil_color: choice("veil_color")
    ring_number: choice("ring_number")
    ring_type: choice("ring_type")
    spore_print_color: choice("spore_print_color")
    population: choice("population") = Field(description="How it grows: alone, in clusters...")
    habitat: choice("habitat")


class MushroomPrediction(BaseModel):
    poisonous_probability: float = Field(description="Probability that it is poisonous (0-1)")
    verdict: Literal["poisonous", "edible"] = Field(
        description="poisonous if poisonous_probability >= threshold"
    )
    threshold: float = Field(description="The decision threshold chosen during training")


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = {}
