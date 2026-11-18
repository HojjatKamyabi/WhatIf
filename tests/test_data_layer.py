"""The data layer (whatif.data) stands on its own: it could be lifted out as a package of its own."""

import ast
from pathlib import Path

import whatif

DATA = Path(whatif.__file__).parent / "data"
# The researcher's modelling choices live outside the data layer and must stay there.
MODELLING_NAMES = ("PARTY_KEYWORDS", "TOPIC_KEYWORDS", "PARTY_FOUNDED", "ATTENTION_ARTICLES", "GOVERNMENTS", "TOPICS")


def whatif_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("whatif"):
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("whatif"))
    return names


def test_the_data_layer_exists_with_one_module_per_source():
    modules = {p.stem for p in DATA.glob("*.py")}
    assert {"adapters", "store", "polls", "elections", "eurostat", "gdelt", "wikipedia"} <= modules


def test_the_data_layer_imports_nothing_from_the_rest_of_whatif():
    for path in DATA.glob("*.py"):
        outside = {m for m in whatif_imports(path) if not m.startswith("whatif.data")}
        assert not outside, f"{path.name} imports {sorted(outside)}"


def test_the_researcher_s_modelling_choices_are_not_in_the_data_layer():
    for path in DATA.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in MODELLING_NAMES:
            assert f"{name}:" not in text and f"{name} =" not in text, f"{name} is defined in {path.name}"


def test_whatifs_gdelt_and_page_view_adapters_carry_whatifs_mapping():
    from whatif.events import GDELT, PAGEVIEWS

    assert GDELT.name == "gdelt-events" and PAGEVIEWS.name == "wikipedia-pageviews"
