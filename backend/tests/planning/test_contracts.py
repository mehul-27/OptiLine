from __future__ import annotations

import math
from dataclasses import replace

import pytest

from optiline.core.planning import (
    PathSample,
    SpeedProfileSample,
    build_speed_profile,
)


def test_invalid_path_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="unit"):
        PathSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        PathSample(0.0, 0.0, 0.0, 1.0, 0.0, math.nan)
    with pytest.raises(ValueError, match="non-negative"):
        PathSample(0.0, 0.0, -1.0, 1.0, 0.0, 0.0)


def test_invalid_path_contracts_are_rejected(circle_path) -> None:
    with pytest.raises(ValueError, match="unique_sample_count"):
        replace(circle_path, samples=circle_path.samples[:-1])
    with pytest.raises(ValueError, match="unique_sample_count"):
        replace(circle_path, unique_sample_count=5)
    with pytest.raises(ValueError, match="closed"):
        replace(circle_path, closed=False)
    with pytest.raises(ValueError, match="schema_version"):
        replace(circle_path, schema_version=2)
    with pytest.raises(ValueError, match="length_m"):
        replace(circle_path, length_m=circle_path.length_m + 1.0)
    duplicate_s = replace(circle_path.samples[1], s_m=0.0)
    with pytest.raises(ValueError, match="increase"):
        replace(
            circle_path, samples=(circle_path.samples[0], duplicate_s, *circle_path.samples[2:])
        )


def test_invalid_speed_profile_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpeedProfileSample(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        SpeedProfileSample(0.0, 10.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="finite"):
        SpeedProfileSample(0.0, 10.0, 0.0, math.nan)


def test_invalid_speed_profile_contracts_are_rejected(
    circle_path,
    planning_limits,
) -> None:
    profile = build_speed_profile(circle_path, planning_limits)
    with pytest.raises(ValueError, match="unique_sample_count"):
        replace(profile, samples=profile.samples[:-1])
    with pytest.raises(ValueError, match="unique_sample_count"):
        replace(profile, unique_sample_count=profile.unique_sample_count + 1)
    with pytest.raises(ValueError, match="closed"):
        replace(profile, closed=False)
    with pytest.raises(ValueError, match="interval acceleration"):
        replace(
            profile,
            interval_longitudinal_accelerations_mps2=profile.interval_longitudinal_accelerations_mps2[
                :-1
            ],
        )
    with pytest.raises(ValueError, match="positive"):
        replace(profile, lap_time_s=0.0)
    with pytest.raises(ValueError, match="finite"):
        replace(
            profile,
            interval_longitudinal_accelerations_mps2=(
                math.nan,
                *profile.interval_longitudinal_accelerations_mps2[1:],
            ),
        )
    last = replace(profile.samples[-1], speed_mps=profile.samples[-1].speed_mps + 1.0)
    with pytest.raises(ValueError, match="start and end speeds"):
        replace(profile, samples=(*profile.samples[:-1], last))
    duplicate_s = replace(profile.samples[1], s_m=0.0)
    with pytest.raises(ValueError, match="increase"):
        replace(
            profile,
            samples=(profile.samples[0], duplicate_s, *profile.samples[2:]),
        )
    with pytest.raises(ValueError, match="final sample"):
        replace(profile, path_length_m=profile.path_length_m + 1.0)
