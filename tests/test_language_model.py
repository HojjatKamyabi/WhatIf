"""Language-model Party Actors: masked prompts, caching, cost, Cutoff Labels and decision modes.

No test calls a real model: a fake client records every prompt and answers from a script.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FIRST_WEEK
from whatif.actors import MENU, EventSummary, Move, StandIn, StateSummary
from whatif.backtest import BacktestConfig, run_backtest
from whatif.boundary import WeeklyBoundary
from whatif.data.elections import parse_results, parse_structure
from whatif.data.polls import PARTIES, TRACKED_PARTIES
from whatif.events import PARTY_KEYWORDS, TOPICS
from whatif.llm import (
    ChatResponse,
    LanguageModelActor,
    Masker,
    ModelConfig,
    OpenAICompatibleClient,
    ResponseCache,
    Usage,
    cutoff_label,
    hindsight_leak_flag,
    load_models,
)
from whatif.poll_average import weekly_poll_average
from whatif.population import build_synthetic_population
from whatif.twin import FIRST_GUESS_REACTIONS, Twin, TwinParameters

ELECTION_FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"
HORIZON = 4
START = FIRST_WEEK + timedelta(weeks=20)
LATEST = np.array([30.0, 20.0, 15.0, 12.0, 8.0, 5.0, 4.0, 6.0])

CONFIG = ModelConfig(
    name="test-model",
    base_url="http://localhost:8000/v1",
    model="open-model-7b",
    quantization="4-bit AWQ",
    training_cutoff=date(2024, 6, 30),
    cutoff_source="https://example.org/model-card",
    api_key_env="WHATIF_TEST_KEY",
    input_price=1.0,  # USD per million tokens
    output_price=4.0,
)


class FakeClient:
    """Records every request; answers with the next scripted reply, or a valid move by default."""

    def __init__(self, replies=None, move="hold"):
        self.replies, self.move, self.requests = list(replies or []), move, []

    def complete(self, messages, seed):
        self.requests.append({"messages": messages, "seed": seed})
        text = self.replies.pop(0) if self.replies else json.dumps({"move": self.move, "reason": "scripted"})
        return ChatResponse(text=text, usage=Usage(prompt_tokens=1000, completion_tokens=50))

    def prompts(self) -> list[str]:
        return [" ".join(m["content"] for m in r["messages"]) for r in self.requests]


def actor(client, tmp_path, party="AfD", seed=0, config=CONFIG, masker=None):
    return LanguageModelActor(
        client=client, config=config, cache=ResponseCache(tmp_path / "cache"),
        masker=masker or Masker(seed=seed), horizon=HORIZON, seed=seed,
    )


def state(party="AfD", events=None) -> StateSummary:
    events = events if events is not None else (
        EventSummary("SPD", "economy", -2.1, 1.3),
        EventSummary("AfD", "migration", -4.0, 2.5),
        EventSummary("Greens", "climate", -1.0, 0.8),
        EventSummary("Left", "social", 1.5, 1.1),
        EventSummary("BSW", "security", -0.5, 0.4),
        EventSummary("CDU/CSU", "economy", -1.2, 1.0),
        EventSummary("FDP", "economy", -0.3, 0.6),
    )
    return StateSummary(
        week=2, party=party, support=dict(zip(PARTIES, LATEST.tolist())), events=tuple(events),
        topic_attention={t: 1.2 for t in TOPICS if t != "other"},
    )


# --- Masked Prompts -------------------------------------------------------------------------------

REAL_NAMES = {"cdu", "csu", "afd", "spd", "greens", "green", "left", "linke", "bsw", "fdp", "germany", "bundestag"}
# Party names and politicians from the keyword table, minus generic words a prompt may use ("party", "social").
GENERIC = {"party", "social", "free", "for", "union", "democrat", "democrats", "democratic", "christian", "alternative"}
REAL_NAMES |= {w for phrases in PARTY_KEYWORDS.values() for p in phrases for w in p.split()} - GENERIC


def test_prompts_contain_no_real_party_or_politician_names_and_no_dates(tmp_path):
    client = FakeClient()
    for party in TRACKED_PARTIES:
        actor(client, tmp_path, party=party).decide(state(party), MENU)
    assert len(client.requests) == len(TRACKED_PARTIES)
    for prompt in client.prompts():
        words = set(re.findall(r"[a-zäöüß]+", prompt.lower()))
        assert not words & REAL_NAMES, words & REAL_NAMES
        assert not re.search(r"\b(19|20)\d\d\b", prompt), "no years"
        assert not re.search(r"\d{4}-\d{2}-\d{2}", prompt), "no dates"
        assert "Party " in prompt and "week 2 of 4" in prompt


def test_party_labels_are_shuffled_per_backtest_seed_and_recorded():
    first, again, other = Masker(seed=1), Masker(seed=1), Masker(seed=2)
    assert first.labels == again.labels
    assert first.labels != other.labels
    assert sorted(first.labels.values()) == [f"Party {c}" for c in "ABCDEFG"]
    assert set(first.labels) == set(TRACKED_PARTIES)
    assert first.describe() == {"labels": first.labels, "weeks": "week N of the Horizon, no dates"}


def test_each_actor_is_told_its_own_credible_topics_under_its_label(tmp_path):
    client = FakeClient()
    masker = Masker(seed=0)
    actor(client, tmp_path, party="AfD", masker=masker).decide(state("AfD"), MENU)
    prompt = client.prompts()[0]
    assert f"You lead the campaign of {masker.labels['AfD']}" in prompt
    assert "credible on: migration, security" in prompt


# --- Answers ----------------------------------------------------------------------------------------


def test_a_valid_answer_becomes_the_move_with_its_reason(tmp_path):
    client = FakeClient(replies=['{"move": "emphasise migration", "reason": "Our topic is hot."}'])
    decision = actor(client, tmp_path).decide(state(), MENU)
    assert decision.move == Move("emphasise", "migration") and decision.reason == "Our topic is hot."


def test_an_invalid_answer_is_retried_once_then_the_party_holds(tmp_path):
    client = FakeClient(replies=["I would rather not say.", '{"move": "dance", "reason": "x"}'])
    a = actor(client, tmp_path)
    decision = a.decide(state(), MENU)
    assert decision.move == Move("hold") and decision.reason == "invalid answer"
    assert len(client.requests) == 2 and a.usage.invalid_answers == 1


def test_a_retry_that_succeeds_counts_no_invalid_answer(tmp_path):
    client = FakeClient(replies=["not json", '{"move": "rebut", "reason": "Answer the attack."}'])
    a = actor(client, tmp_path)
    assert a.decide(state(), MENU).move == Move("rebut")
    assert a.usage.invalid_answers == 0 and a.usage.retries == 1


# --- Cache and cost --------------------------------------------------------------------------------


def test_the_same_prompt_and_seed_reuse_the_cached_answer_even_after_a_restart(tmp_path):
    client = FakeClient()
    actor(client, tmp_path).decide(state(), MENU)
    rerun = actor(client, tmp_path)  # a new process: the cache is on disk
    rerun.decide(state(), MENU)
    assert len(client.requests) == 1
    assert rerun.usage.calls == 0 and rerun.usage.cache_hits == 1


@pytest.mark.parametrize(
    "change", [{"model": "other-model"}, {"quantization": "full precision"}, {"base_url": "https://api.example.org/v1"}]
)
def test_changing_model_quantization_or_endpoint_misses_the_cache(tmp_path, change):
    from dataclasses import replace

    client = FakeClient()
    actor(client, tmp_path).decide(state(), MENU)
    actor(client, tmp_path, config=replace(CONFIG, **change)).decide(state(), MENU)
    assert len(client.requests) == 2


def test_a_different_seed_misses_the_cache(tmp_path):
    client = FakeClient()
    masker = Masker(seed=0)
    actor(client, tmp_path, seed=0, masker=masker).decide(state(), MENU)
    actor(client, tmp_path, seed=1, masker=masker).decide(state(), MENU)
    assert len(client.requests) == 2


def test_cost_is_tokens_times_the_configured_prices_and_cached_answers_cost_nothing(tmp_path):
    client = FakeClient()
    a = actor(client, tmp_path)
    a.decide(state(), MENU)
    a.decide(state(), MENU)  # cached
    # 1,000 prompt tokens at $1/M plus 50 completion tokens at $4/M, once.
    assert a.usage.cost_usd == pytest.approx(0.001 + 0.0002)
    assert (a.usage.calls, a.usage.cache_hits, a.usage.prompt_tokens, a.usage.completion_tokens) == (1, 1, 1000, 50)


# --- Configuration and the HTTP client ---------------------------------------------------------------


def test_models_are_read_from_a_config_file_without_any_key(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[model.local]\nbase_url = "http://localhost:8000/v1"\nmodel = "open-model-7b"\n'
        'quantization = "4-bit AWQ"\ntraining_cutoff = 2024-06-30\ncutoff_source = "https://example.org/card"\n'
        'api_key_env = "MY_KEY"\ninput_price = 0.0\noutput_price = 0.0\n',
        encoding="utf-8",
    )
    model = load_models(path)["local"]
    assert (model.model, model.quantization, model.training_cutoff) == ("open-model-7b", "4-bit AWQ", date(2024, 6, 30))
    assert model.api_key_env == "MY_KEY" and model.temperature == 0.7
    assert "key" not in {k for k in vars(model) if k != "api_key_env"}


def test_the_client_posts_an_openai_compatible_chat_request_with_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("WHATIF_TEST_KEY", "test-value")
    sent = {}

    def post(url, body, headers):
        sent.update(url=url, body=json.loads(body), headers=headers)
        return json.dumps({
            "choices": [{"message": {"content": '{"move": "hold", "reason": "r"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }).encode()

    response = OpenAICompatibleClient(CONFIG, post=post).complete([{"role": "user", "content": "hi"}], seed=7)
    assert sent["url"] == "http://localhost:8000/v1/chat/completions"
    assert sent["body"]["model"] == "open-model-7b" and sent["body"]["seed"] == 7
    assert sent["body"]["temperature"] == 0.7
    assert sent["headers"]["Authorization"] == "Bearer test-value"
    assert response.text == '{"move": "hold", "reason": "r"}' and response.usage.prompt_tokens == 12


def test_a_missing_key_variable_is_reported_by_name_without_calling(monkeypatch):
    monkeypatch.delenv("WHATIF_TEST_KEY", raising=False)
    client = OpenAICompatibleClient(CONFIG, post=lambda *a: pytest.fail("called without a key"))
    with pytest.raises(RuntimeError, match="WHATIF_TEST_KEY"):
        client.complete([{"role": "user", "content": "hi"}], seed=0)


# --- Cutoff Labels and the leak flag ------------------------------------------------------------------

CUTOFF = date(2024, 6, 30)


@pytest.mark.parametrize(
    "first, last, label",
    [
        (date(2023, 1, 2), date(2023, 3, 26), "Before"),  # ends before the cutoff
        (date(2024, 6, 3), date(2024, 8, 25), "Straddling"),  # crosses it
        (date(2024, 8, 5), date(2024, 10, 27), "Straddling"),  # after it, but within the 3-month margin
        (date(2024, 10, 7), date(2024, 12, 29), "After"),  # starts more than 3 months after it
    ],
)
def test_cutoff_labels_follow_the_whole_window_and_the_safety_margin(first, last, label):
    assert cutoff_label(first, last, CUTOFF, margin_months=3) == label


def test_a_possible_hindsight_leak_is_flagged_only_when_just_the_language_model_beats_the_statistical_baseline():
    assert hindsight_leak_flag(llm=0.40, stand_in=0.55, statistical=0.50)
    assert not hindsight_leak_flag(llm=0.40, stand_in=0.45, statistical=0.50)  # both beat it
    assert not hindsight_leak_flag(llm=0.60, stand_in=0.55, statistical=0.50)  # neither does


# --- Decision modes in a Backtest --------------------------------------------------------------------


@pytest.fixture(scope="module")
def population():
    results = parse_results((ELECTION_FIXTURES / "kerg2_2025.csv").read_bytes())
    structure = parse_structure((ELECTION_FIXTURES / "strukturdaten_2025.csv").read_bytes())
    return build_synthetic_population(results, structure, scale=100, seed=0)


def llm_twin(population, client, tmp_path, mode, seed=0):
    masker = Masker(seed=seed)
    actors = {
        p: LanguageModelActor(client=client, config=CONFIG, cache=ResponseCache(tmp_path / "cache"),
                              masker=masker, horizon=HORIZON, seed=seed)
        for p in TRACKED_PARTIES
    }
    boundary = WeeklyBoundary.quiet(HORIZON, government=frozenset({"SPD", "Greens"}))
    return Twin(population, TwinParameters(**FIRST_GUESS_REACTIONS), seed=seed, boundary=boundary,
                actors=actors, decision_mode=mode, name=f"twin:{mode}")


def test_shared_actor_decisions_make_horizon_times_parties_calls_whatever_the_runs(poll_series, population, tmp_path):
    client = FakeClient(move="emphasise economy")
    twin = llm_twin(population, client, tmp_path, "shared")
    result = run_backtest(weekly_poll_average(poll_series), BacktestConfig(start_week=START, horizon=HORIZON, runs=50), [twin])
    assert len(client.requests) == HORIZON * len(TRACKED_PARTIES)
    assert len(result.logs["twin:shared"]) == HORIZON * len(TRACKED_PARTIES)


def test_per_run_decisions_call_the_model_for_every_run_with_its_own_seed(poll_series, population, tmp_path):
    client = FakeClient(move="emphasise economy")
    twin = llm_twin(population, client, tmp_path, "per-run")
    result = run_backtest(weekly_poll_average(poll_series), BacktestConfig(start_week=START, horizon=HORIZON, runs=3), [twin])
    assert len(client.requests) == 3 * HORIZON * len(TRACKED_PARTIES)
    assert {r["seed"] for r in client.requests} == {0, 1, 2}
    log = result.logs["twin:per-run"]
    assert {d["run"] for d in log} == {0, 1, 2}


def test_stand_ins_run_in_either_decision_mode_with_the_same_moves(poll_series, population):
    boundary = WeeklyBoundary.quiet(HORIZON, government=frozenset({"SPD", "Greens"}))
    twins = [
        Twin(population, TwinParameters(**FIRST_GUESS_REACTIONS), seed=0, boundary=boundary,
             actors={p: StandIn() for p in TRACKED_PARTIES}, decision_mode=mode, name=mode)
        for mode in ("shared", "per-run")
    ]
    result = run_backtest(
        weekly_poll_average(poll_series), BacktestConfig(start_week=START, horizon=HORIZON, runs=5), twins,
        shared_noise=True,
    )
    shared = [d["move"] for d in result.logs["shared"]]
    assert [d["move"] for d in result.logs["per-run"] if d["run"] == 0] == shared
    assert result.scores["shared"].quantiles == result.scores["per-run"].quantiles  # deterministic Stand-ins
