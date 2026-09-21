from src.data.typologies import EVALUATED_TYPOLOGIES, Typology, normalize_typology


def test_all_eight_typologies_present():
    names = {t.value for t in EVALUATED_TYPOLOGIES}
    assert names == {
        "fan_out", "fan_in", "cycle", "gather_scatter",
        "scatter_gather", "stack", "bipartite", "random",
    }
    assert Typology.UNCLASSIFIED not in EVALUATED_TYPOLOGIES


def test_normalize_typology_known_label():
    assert normalize_typology("FAN-OUT") == Typology.FAN_OUT
    assert normalize_typology("fan-out") == Typology.FAN_OUT  # case-insensitive
    assert normalize_typology("  CYCLE  ") == Typology.CYCLE  # whitespace-tolerant


def test_normalize_typology_unknown_label_falls_back():
    assert normalize_typology("SOMETHING-ELSE") == Typology.UNCLASSIFIED
