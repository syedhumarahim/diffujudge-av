from __future__ import annotations

import pytest

from diffujudge.taxonomy import BEHAVIOR_CATEGORIES, classify_label


def test_taxonomy_codes_unique():
    codes = [c.code for c in BEHAVIOR_CATEGORIES]
    assert len(codes) == len(set(codes))


def test_classify_known():
    cat = classify_label("cut_in")
    assert cat.is_safety_critical
    assert cat.nhtsa_scenario == 22


def test_classify_unknown():
    with pytest.raises(KeyError):
        classify_label("not-a-real-code")
