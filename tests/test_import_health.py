"""Every module must import, and the timing helpers the server needs must exist.

A half-written edit or a stale bytecode cache used to surface as

    cannot import name 'normalize_note_text' from 'maichart.timing'

i.e. the WebUI simply stops working, with nothing in the test suite to catch it.
Importing the whole package here turns that class of failure into a fast, local
test result instead of a broken server.
"""

import importlib
import pkgutil

import maichart


def test_every_module_imports():
    failures = []
    imported = 0
    for module in pkgutil.walk_packages(maichart.__path__, "maichart."):
        name = module.name
        if "scripts" in name or name.endswith("__main__"):
            continue
        try:
            importlib.import_module(name)
            imported += 1
        except Exception as exc:  # noqa: BLE001 - the failure list is the report
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert imported > 30, "the sweep should cover the package"
    assert not failures, "modules that fail to import:\n" + "\n".join(failures)


def test_timing_helpers_are_importable_from_their_documented_home():
    """``maichart.timing`` owns the note-text normalizer; callers rely on it."""

    from maichart.timing import normalize_note_text, tokenize_difficulty_timing

    # It collapses repeated identical touch areas (``EE`` -> ``E``) and is
    # idempotent; ordinary chords and positioned touches are untouched.
    assert normalize_note_text("EE") == "E"
    assert normalize_note_text("36/E8") == "36/E8"
    assert normalize_note_text(normalize_note_text("EE/B8")) == normalize_note_text("EE/B8")
    assert callable(tokenize_difficulty_timing)

    # event_timing re-exports it for the event pipeline.
    from maichart.event_timing import parse_event_chart  # noqa: F401
    from maichart import event_timing

    assert event_timing.normalize_note_text is normalize_note_text


def test_every_cli_can_print_its_help():
    """``--help`` must not crash, and help text must escape literal percents.

    A help string containing ``~6% of notes`` makes argparse interpolate it as a
    printf spec (``%o``), so ``--help`` raised

        TypeError: %o format: an integer is required, not dict

    and printed a traceback instead of the usage.  Formatting every parser here
    catches the whole class of bug, in every CLI, without spawning processes.
    """

    import importlib

    parsers = [
        ("maichart.training.train_events", "build_parser"),
        ("maichart.generation.event_cli", "build_parser"),
        ("maichart.generation.acceptance", "build_parser"),
    ]
    rendered = {}
    for module_name, factory in parsers:
        module = importlib.import_module(module_name)
        builder = getattr(module, factory, None)
        if builder is None:
            continue
        text = builder().format_help()
        assert len(text) > 500, f"{module_name} help looks empty"
        rendered[module_name] = text

    assert rendered, "no CLI parser was exercised"
    trainer = rendered.get("maichart.training.train_events", "")
    # Sampling controls must remain independently discoverable after splitting
    # slide and touch sampling. Corpus-specific percentage claims are not a
    # public CLI contract.
    for flag in ("--cuda-memory-budget-gb", "--cuda-cache-flush-batches", "--prefetch-batches",
                 "--slide-oversample", "--touch-oversample"):
        assert flag in trainer, f"{flag} missing from the trainer's help"
    assert "%o" not in trainer

