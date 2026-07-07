"""Generate sample property records for UI demo.

This script creates data/sample_properties.csv based on the project features.
It is used only for demonstration/testing of the UI.

Generated columns:
- Area
- Bedrooms
- Bathrooms
- Floor
- City
- Furnished
- Rent
- ImagePath (fake placeholder)

No ANN training is performed here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class SampleProperty:
    area: float
    bedrooms: int
    bathrooms: int
    floor: int
    city: str
    furnished: str
    rent: float
    image_path: str


def generate_samples(n: int = 100, seed: int = 42) -> List[SampleProperty]:
    rng = np.random.default_rng(seed)

    cities = ["Hanoi", "Da Nang", "Hochiminh"]
    furnished_choices = ["Yes", "No"]

    samples: List[SampleProperty] = []

    # Simple realistic rent model:
    # rent ~ area * city_factor + bedrooms*250 + bathrooms*180 + floor_effect + furnished_effect + noise
    city_factor = {"Hanoi": 18.5, "Da Nang": 14.8, "Hochiminh": 21.2}
    furnished_factor = {"Yes": 450, "No": 0}
    floor_factor = lambda f: (f - 5) * 40

    for _ in range(n):
        area = float(rng.integers(45, 1050))

        bedrooms = int(np.clip(rng.integers(1, 5), 1, 4))
        bathrooms = int(np.clip(rng.integers(1, 4), 1, 3))

        # Align floor to a common range
        floor = int(rng.integers(1, 15))

        city = str(rng.choice(cities))
        furnished = str(rng.choice(furnished_choices))

        base = area * city_factor[city] / 10.0  # scale down
        rent = (
            base
            + bedrooms * 250
            + bathrooms * 180
            + floor_factor(floor)
            + furnished_factor[furnished]
            + float(rng.normal(0, 250))
        )

        rent = float(max(800, rent))

        # Fake image placeholder name
        img_id = int(rng.integers(1, 1000))
        image_path = f"assets/building_{img_id:03d}.png"

        samples.append(
            SampleProperty(
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                floor=floor,
                city=city,
                furnished=furnished,
                rent=rent,
                image_path=image_path,
            )
        )

    # Keep the CSV stable by sorting by rent
    samples.sort(key=lambda s: s.rent)
    return samples


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_path = root / "data" / "sample_properties.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(n=100)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Area", "Bedrooms", "Bathrooms", "Floor", "City", "Furnished", "Rent", "ImagePath"]
        )
        for s in samples:
            writer.writerow(
                [s.area, s.bedrooms, s.bathrooms, s.floor, s.city, s.furnished, s.rent, s.image_path]
            )


if __name__ == "__main__":
    main()

