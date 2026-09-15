import pytest

from aster_hpo import Choice, Float, Int, sample_configs


def test_sampling_reproducible_independent_of_mapping_order():
    space = {
        "lr": Float(1e-5, 0.1, log=True),
        "width": Int(8, 64),
        "activation": Choice(("a", "b")),
    }
    a = sample_configs(space, count=100, seed=5)
    assert a == sample_configs(dict(reversed(list(space.items()))), count=100, seed=5)
    assert a != sample_configs(space, count=100, seed=6)
    assert all(1e-5 <= c["lr"] <= 0.1 and 8 <= c["width"] <= 64 for c in a)
    assert {c["activation"] for c in a} == {"a", "b"}


@pytest.mark.parametrize("bounds", [(2, 1), (1, 1), (0, float("inf"))])
def test_invalid_float(bounds):
    with pytest.raises(ValueError):
        Float(*bounds)


def test_invalid_log():
    with pytest.raises(ValueError):
        Float(0, 1, log=True)


def test_empty_choice():
    with pytest.raises(ValueError):
        Choice(())


def test_empty_space():
    with pytest.raises(ValueError):
        sample_configs({}, count=2)
