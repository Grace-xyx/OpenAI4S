"""Offline contracts for the Skill-suggestion evaluation (W2-C / plan §7)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import openai4s.store as store_mod
from harness.evals import judgment_skills as ev
from openai4s.config import Config
from openai4s.storage.migrations import SCHEMA_VERSION

CASES_PATH = ev.CASES_PATH
LOCK_PATH = ev.LOCK_PATH
TERMS_PATH = ev.TERMS_PATH


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return ev.load_cases(CASES_PATH)


@pytest.fixture
def eval_cfg(tmp_path: Path) -> Config:
    # Explicit, not resolved from the environment: the loader opens
    # `cfg.db_path` on every search. This was a module-scoped `Config()`, set
    # up before conftest's per-test OPENAI4S_DATA_DIR redirect, so it read the
    # developer's real ~/.openai4s database.
    return Config(data_dir=tmp_path / "eval-data")


@pytest.fixture
def skill_names(eval_cfg: Config) -> set[str]:
    return ev.known_skill_names(eval_cfg)


@pytest.fixture
def opened_stores(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every database path a `Store` is constructed for while the test runs."""

    opened: list[Path] = []
    original = store_mod.Store.__init__

    def recording_init(self, db_path, *args, **kwargs):
        opened.append(Path(db_path))
        original(self, db_path, *args, **kwargs)

    monkeypatch.setattr(store_mod.Store, "__init__", recording_init)
    return opened


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


def test_dataset_schema_and_counts(cases: list[dict]) -> None:
    assert 180 <= len(cases) <= 220
    zh = [c for c in cases if c["lang"] == "zh"]
    en = [c for c in cases if c["lang"] == "en"]
    assert 90 <= len(zh) <= 110
    assert 90 <= len(en) <= 110
    no_skill = [c for c in cases if c["category"] == "no_skill"]
    assert len(no_skill) / len(cases) >= 0.20
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert set(case) == set(ev.CASE_FIELDS)
        assert case["lang"] in ev.LANGS
        assert case["category"] in ev.CATEGORIES
        assert case["split"] in ev.SPLITS
        assert isinstance(case["query"], str) and case["query"].strip()
        assert case["query"] == case["query"].strip()
        assert isinstance(case["notes"], str)
        assert isinstance(case["gold"], list)
        assert 0 <= len(case["gold"]) <= 3
        if case["category"] == "no_skill":
            assert case["gold"] == []


def test_every_lang_category_split_stratum_is_nonempty(cases: list[dict]) -> None:
    counts = ev.stratum_counts(cases)
    for lang in ev.LANGS:
        for category in ev.CATEGORIES:
            for split in ev.SPLITS:
                key = f"{lang}:{category}:{split}"
                assert counts.get(key, 0) >= 1, key


def test_gold_names_exist_in_the_loader(
    cases: list[dict], skill_names: set[str]
) -> None:
    missing = [
        (case["id"], name)
        for case in cases
        for name in case["gold"]
        if name not in skill_names
    ]
    assert missing == []


def test_lock_file_matches_the_frozen_test_split(cases: list[dict]) -> None:
    lock = ev.load_lock(LOCK_PATH)
    assert lock["algorithm"] == "sha256"
    assert lock["sha256"] == ev.test_set_sha256(cases)
    assert lock["n_test"] == sum(1 for case in cases if case["split"] == "test")
    mutated = [dict(case) for case in cases]
    for row in mutated:
        if row["split"] == "test":
            row["query"] = row["query"] + " [mutated]"
            break
    else:
        raise AssertionError("no test case to mutate")
    assert ev.test_set_sha256(mutated) != lock["sha256"]


def test_b2_glossary_is_dev_only_and_capped(cases: list[dict]) -> None:
    payload = json.loads(TERMS_PATH.read_text(encoding="utf-8"))
    assert "_construction" in payload
    assert (
        "development split" in payload["_construction"].lower()
        or "split=dev" in payload["_construction"]
    )
    terms = ev.load_glossary(TERMS_PATH)
    assert 1 <= len(terms) <= 300
    blob = "\n".join(
        case["query"]
        for case in cases
        if case["split"] == "dev" and case["lang"] == "zh"
    )
    test_blob = "\n".join(
        case["query"]
        for case in cases
        if case["split"] == "test" and case["lang"] == "zh"
    )
    for item in terms:
        assert item["zh"] in blob
        # Presence in the test blob is allowed only as a side effect of also
        # appearing in dev. The builder never read the test split.
        assert item["zh"] in blob or item["zh"] not in test_blob


def test_expand_query_b2_is_zh_only() -> None:
    terms = [{"zh": "单细胞", "en": "single-cell scrna-seq"}]
    zh = ev.expand_query_b2("帮我看看这批单细胞数据", "zh", terms)
    assert "single-cell" in zh
    assert zh.startswith("帮我看看这批单细胞数据")
    en = ev.expand_query_b2("look at this 单细胞 dataset", "en", terms)
    assert en == "look at this 单细胞 dataset"
    untouched = ev.expand_query_b2("没有这个词", "zh", terms)
    assert untouched == "没有这个词"


def test_case_indicators_edges() -> None:
    empty = ev.case_indicators([], [])
    assert empty["abstention"] == 1
    assert empty["unnecessary_recommendation"] == 0
    assert empty["top1"] is None
    assert empty["top3_recall"] is None
    assert empty["error_recommendation"] is None

    extra = ev.case_indicators([], ["alphafold2"])
    assert extra["abstention"] == 0
    assert extra["unnecessary_recommendation"] == 1
    assert extra["top1"] is None

    hit = ev.case_indicators(["alphafold2"], ["alphafold2", "boltz"])
    assert hit["top1"] == 1
    assert hit["top3_recall"] == 1
    assert hit["error_recommendation"] == 1
    assert hit["abstention"] == 0
    assert hit["unnecessary_recommendation"] is None

    subset = ev.case_indicators(["alphafold2", "boltz"], ["alphafold2"])
    assert subset["top1"] == 1
    assert subset["top3_recall"] == 1
    assert subset["error_recommendation"] == 0

    miss = ev.case_indicators(["alphafold2"], ["diffdock"])
    assert miss["top1"] == 0
    assert miss["top3_recall"] == 0
    assert miss["error_recommendation"] == 1

    abstain_gold = ev.case_indicators(["alphafold2"], [])
    assert abstain_gold["top1"] == 0
    assert abstain_gold["top3_recall"] == 0
    assert abstain_gold["error_recommendation"] == 0
    assert abstain_gold["abstention"] == 1

    reorder = ev.case_indicators(["a", "b"], ["b", "a"])
    assert reorder["top1"] == 1
    assert reorder["top3_recall"] == 1
    assert reorder["error_recommendation"] == 0


def test_score_predictions_and_bootstrap_bounds() -> None:
    cases = [
        {
            "id": "c1",
            "lang": "en",
            "category": "curated",
            "gold": ["alphafold2"],
            "query": "x",
            "notes": "",
            "split": "dev",
        },
        {
            "id": "c2",
            "lang": "zh",
            "category": "no_skill",
            "gold": [],
            "query": "y",
            "notes": "",
            "split": "dev",
        },
        {
            "id": "c3",
            "lang": "en",
            "category": "curated",
            "gold": ["boltz"],
            "query": "z",
            "notes": "",
            "split": "dev",
        },
    ]
    report = ev.score_predictions(
        cases,
        {"c1": ["alphafold2"], "c2": ["literature-review"], "c3": []},
        extra={
            "c1": {"requests": 0, "latency_ms": 10, "input_tokens": 0},
            "c2": {"requests": 0, "latency_ms": 20, "input_tokens": 0},
            "c3": {"requests": 1, "latency_ms": 30, "input_tokens": 100},
        },
    )
    overall = report["overall"]
    assert overall["n"] == 3
    assert overall["top1_accuracy"]["n"] == 2
    assert overall["top1_accuracy"]["value"] == 0.5
    assert overall["top3_recall"]["value"] == 0.5
    assert overall["unnecessary_recommendation_rate"]["value"] == 1.0
    assert overall["abstention_rate"]["value"] == pytest.approx(1 / 3)
    assert overall["input_tokens"] == 100
    assert overall["cost_usd"] == pytest.approx(
        100 / 1_000_000 * ev.INPUT_TOKEN_USD_PER_MILLION
    )
    lo, hi = overall["top1_accuracy"]["ci95"]
    assert lo is not None and hi is not None
    assert lo <= overall["top1_accuracy"]["value"] <= hi
    assert report["by_lang"]["zh"]["n"] == 1
    assert (
        report["by_category"]["no_skill"]["unnecessary_recommendation_rate"]["value"]
        == 1.0
    )


def test_bootstrap_empty_and_degenerate() -> None:
    assert ev.bootstrap_mean_ci([]) == (None, None)
    assert ev.bootstrap_mean_ci([1.0]) == (1.0, 1.0)
    lo, hi = ev.bootstrap_mean_ci([0.0, 0.0, 0.0])
    assert lo == 0.0 and hi == 0.0
    lo, hi = ev.bootstrap_mean_ci([0.0, 1.0] * 10)
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_split_test_refuses_without_matching_frozen_thresholds(
    cases: list[dict],
) -> None:
    with pytest.raises(SystemExit, match="frozen-thresholds"):
        ev.build_report(
            split="test",
            systems=["B0"],
            cases=cases,
            live=False,
            frozen_thresholds=None,
        )
    with pytest.raises(SystemExit, match="TEMPLATE_VERSION|frozen-thresholds"):
        ev.build_report(
            split="test",
            systems=["B0"],
            cases=cases,
            live=False,
            frozen_thresholds="not-a-real-template-version",
        )


def test_live_systems_require_the_live_flag(cases: list[dict]) -> None:
    with pytest.raises(SystemExit, match="--live"):
        ev.build_report(
            split="dev",
            systems=["B1"],
            cases=cases[:1],
            live=False,
            frozen_thresholds=None,
        )


def test_parse_systems_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ev.parse_systems("B0,B9")
    assert ev.parse_systems("B0,B2,B0") == ["B0", "B2"]


def test_assign_splits_is_deterministic() -> None:
    raw = [
        {
            "id": f"{lang}-{category}-{i:02d}",
            "lang": lang,
            "category": category,
            "query": "q",
            "gold": [],
            "notes": "",
        }
        for lang in ev.LANGS
        for category in ev.CATEGORIES
        for i in range(4)
    ]
    first = ev.assign_splits(raw, seed=ev.SPLIT_SEED)
    second = ev.assign_splits(raw, seed=ev.SPLIT_SEED)
    assert [c["split"] for c in first] == [c["split"] for c in second]
    counts = ev.stratum_counts(first)
    for lang in ev.LANGS:
        for category in ev.CATEGORIES:
            assert counts[f"{lang}:{category}:dev"] == 2
            assert counts[f"{lang}:{category}:test"] == 2


@pytest.mark.slow
def test_b0_and_b2_run_offline_on_the_dev_split(
    cases: list[dict], eval_cfg: Config
) -> None:
    dev = [case for case in cases if case["split"] == "dev"]
    assert dev
    terms = ev.load_glossary(TERMS_PATH)
    for system in ("B0", "B2"):
        report = ev.evaluate_system(system, dev, terms=terms, cfg=eval_cfg)
        assert report["overall"]["n"] == len(dev)
        assert report["system"] == system
        errors = [row["error"] for row in report["cases"] if row.get("error")]
        assert errors == []
        assert all(len(row["pred"]) <= 3 for row in report["cases"])
        zh = report["by_lang"]["zh"]
        en = report["by_lang"]["en"]
        assert zh["n"] + en["n"] == len(dev)
        # B0/B2 do not call a judgment backend.
        assert report["overall"]["input_tokens"] == 0
        assert report["overall"]["cost_usd"] == 0.0


def _plant_newer_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at a ~/.openai4s that a newer build already migrated.

    This is the developer machine the offline suite failed on: opening that
    database raises FutureSchemaError, and opening an older one would migrate
    it for good.
    """

    home = tmp_path / "home"
    database = home / ".openai4s" / "openai4s.db"
    database.parent.mkdir(parents=True)
    conn = sqlite3.connect(database)
    try:
        conn.execute("CREATE TABLE sentinel (x)")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_explicit_cfg_keeps_the_eval_out_of_the_home_data_dir(
    cases: list[dict],
    eval_cfg: Config,
    opened_stores: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _plant_newer_home(tmp_path, monkeypatch)
    # The window a module-scoped fixture runs in: conftest's per-test redirect
    # is not applied yet, so a default Config() resolves the home data dir.
    monkeypatch.delenv("OPENAI4S_DATA_DIR", raising=False)
    assert Config().data_dir == home / ".openai4s"
    before = _tree(home)

    assert ev.known_skill_names(eval_cfg)
    dev = [case for case in cases if case["split"] == "dev"][:8]
    report = ev.evaluate_system("B0", dev, cfg=eval_cfg)

    assert [row["error"] for row in report["cases"] if row.get("error")] == []
    assert opened_stores, "B0 search never opened a Store; the probe saw nothing"
    assert all(path.is_relative_to(eval_cfg.data_dir) for path in opened_stores)
    assert _tree(home) == before


def test_default_cfg_is_resolved_per_call_not_pinned_by_the_first(
    cases: list[dict],
    opened_stores: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = _plant_newer_home(tmp_path, monkeypatch)
    before = _tree(home)
    dev = [case for case in cases if case["split"] == "dev"][:4]
    # Two data dirs in one process, as two tests on one xdist worker see them.
    # The loader used to be resolved once and cached for the process, so the
    # second run searched the first run's database -- or, when the first call
    # came from outside the redirect, the home one.
    for name in ("first", "second"):
        data_dir = tmp_path / name
        monkeypatch.setenv("OPENAI4S_DATA_DIR", str(data_dir))
        opened_stores.clear()

        report = ev.evaluate_system("B0", dev)

        assert [row["error"] for row in report["cases"] if row.get("error")] == []
        assert opened_stores, f"{name}: search never opened this run's Store"
        assert all(path.is_relative_to(data_dir) for path in opened_stores), name
    assert _tree(home) == before
