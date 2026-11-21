"""Language-model Party Actors, through any OpenAI-compatible chat endpoint.

The same Backtest runs on a self-hosted server (vLLM, SGLang) or a hosted API: a model is a base
URL and a model name, plus what the endpoint can't report (quantization, Training Cutoff, prices).
No provider SDK is used, only a plain HTTP request.

Every prompt is a Masked Prompt: parties are "Party A" to "Party G" (shuffled per Backtest),
time is "week N of the Horizon", and nothing of the real news reaches the model. Answers are
cached on disk by prompt, seed, model, quantization and endpoint, so a rerun makes no calls.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np

from whatif.actors import ISSUE_OWNERSHIP, MENU, Decision, Move, StateSummary
from whatif.data.adapters import USER_AGENT
from whatif.data.polls import TRACKED_PARTIES

# --- Models -------------------------------------------------------------------------------------


class ModelConfigError(ValueError):
    """A model that is missing from, or badly written in, the models file."""



@dataclass(frozen=True)
class ModelConfig:
    """One model behind one OpenAI-compatible endpoint. Holds the name of the key's variable, never the key."""

    name: str
    base_url: str  # e.g. http://localhost:8000/v1
    model: str
    quantization: str  # what the endpoint serves, e.g. "full precision" or "4-bit AWQ"
    training_cutoff: date  # the published Training Cutoff
    cutoff_source: str  # where it was published
    api_key_env: str | None = None  # the environment variable holding the key; None for none
    input_price: float = 0.0  # USD per million prompt tokens (0 for a self-hosted server)
    output_price: float = 0.0  # USD per million completion tokens
    temperature: float = 0.7
    max_tokens: int = 300

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "quantization": self.quantization,
            "endpoint": self.base_url,
            "training_cutoff": self.training_cutoff.isoformat(),
            "cutoff_source": self.cutoff_source,
            "temperature": self.temperature,
        }


_MODEL_KEYS = {f for f in ModelConfig.__dataclass_fields__ if f != "name"}


def load_models(path: Path) -> dict[str, ModelConfig]:
    """Every `[model.NAME]` in a TOML file."""
    if not path.is_file():
        raise FileNotFoundError(f"No model config at {path}. Copy models.example.toml to {path.name} and fill it in.")
    table = tomllib.loads(path.read_text(encoding="utf-8"))
    models = {}
    for name, entry in table.get("model", {}).items():
        unknown = set(entry) - _MODEL_KEYS
        if unknown:
            raise ValueError(f"{path}: model {name!r} has unknown field(s) {sorted(unknown)}.")
        models[name] = ModelConfig(name=name, **entry)
    return models


# --- The client -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: Usage


class ChatClient(Protocol):
    def complete(self, messages: list[dict], seed: int) -> ChatResponse: ...


def _post(url: str, body: bytes, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


class OpenAICompatibleClient:
    """POST {base_url}/chat/completions, the request every OpenAI-compatible server accepts."""

    def __init__(self, config: ModelConfig, post: Callable[[str, bytes, dict], bytes] | None = None):
        self.config, self._post = config, post if post is not None else _post

    def complete(self, messages: list[dict], seed: int) -> ChatResponse:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise RuntimeError(
                    f"Model {self.config.name!r} needs its key in the environment variable "
                    f"{self.config.api_key_env}, which is not set."
                )
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": self.config.model,
            "messages": messages,
            "seed": seed,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        raw = self._post(f"{self.config.base_url.rstrip('/')}/chat/completions", json.dumps(body).encode(), headers)
        reply = json.loads(raw)
        usage = reply.get("usage") or {}
        return ChatResponse(
            text=reply["choices"][0]["message"]["content"] or "",
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
        )


# --- Cache and cost --------------------------------------------------------------------------------


class ResponseCache:
    """Answers on disk, one file per key: prompt, seed, model, quantization, endpoint and temperature."""

    def __init__(self, directory: Path):
        self.directory = directory

    @staticmethod
    def key(messages: list[dict], seed: int, config: ModelConfig) -> str:
        identity = {
            "messages": messages,
            "seed": seed,
            "model": config.model,
            "quantization": config.quantization,
            "endpoint": config.base_url.rstrip("/"),
            "temperature": config.temperature,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> ChatResponse | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        stored = json.loads(path.read_text(encoding="utf-8"))
        return ChatResponse(stored["text"], Usage(**stored["usage"]))

    def put(self, key: str, response: ChatResponse) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"text": response.text, "usage": vars(response.usage)}
        (self.directory / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


@dataclass
class UsageLedger:
    """What the language model cost in one Backtest. Cached answers cost nothing."""

    input_price: float = 0.0
    output_price: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    invalid_answers: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * self.input_price + self.completion_tokens * self.output_price) / 1e6

    def describe(self) -> dict:
        return {
            "calls": self.calls, "cache_hits": self.cache_hits, "retries": self.retries,
            "invalid_answers": self.invalid_answers, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost_usd, 6),
        }


# --- Masking ----------------------------------------------------------------------------------------


class Masker:
    """Neutral labels for the parties, shuffled per Backtest from its seed."""

    def __init__(self, seed: int):
        order = np.random.default_rng([seed, 9]).permutation(len(TRACKED_PARTIES))
        self.labels = {TRACKED_PARTIES[i]: f"Party {chr(ord('A') + rank)}" for rank, i in enumerate(order)}

    def label(self, party: str) -> str:
        return self.labels.get(party, "Others")

    def describe(self) -> dict:
        return {"labels": dict(self.labels), "weeks": "week N of the Horizon, no dates"}


_SYSTEM = (
    "You are the campaign strategist of one party in a simulated multi-party democracy. Parties are "
    "known only by letters, and time only by week number. Each week you choose exactly one move from "
    "a fixed menu. Answer only with a JSON object of the form "
    '{"move": "<one move from the menu, written exactly>", "reason": "<one short sentence>"}.'
)


def masked_prompt(state: StateSummary, masker: Masker, horizon: int, menu: tuple[Move, ...]) -> list[dict]:
    """The Masked Prompt for one Party Actor's weekly decision."""
    me = masker.label(state.party)
    support = ", ".join(
        f"{masker.label(p)} {state.support[p]:.1f}"
        for p in sorted(state.support, key=lambda p: (p not in masker.labels, masker.label(p)))
    )
    events = [  # coverage under 5% of a usual week says nothing and only costs tokens
        f"- {masker.label(e.party)}, {e.topic}, tone {e.tone:+.1f}, size {e.size:.1f}"
        for e in sorted(state.events, key=lambda e: (masker.label(e.party), e.topic))
        if e.size >= 0.05
    ]
    attention = ", ".join(f"{topic} {value:.2f}" for topic, value in state.topic_attention.items())
    credible = ", ".join(ISSUE_OWNERSHIP.get(state.party, ())) or "none in particular"
    lines = [
        f"You lead the campaign of {me}. It is week {state.week} of {horizon}.",
        f"Simulated national support (%) at the end of last week: {support}.",
        "This week's news events (party, topic, tone from -10 very negative to +10 very positive, "
        "size relative to a usual week for that party):",
        *(events or ["- none recorded"]),
        f"Public attention to each topic relative to usual (1.00 = usual): {attention}.",
        f"Topics your party is credible on: {credible}.",
        "Moves: 'emphasise <topic>' puts that topic at the centre of your campaign this week; "
        "'rebut' answers this week's negative coverage of your party; 'hold' changes nothing.",
        f"Menu: {', '.join(map(str, menu))}.",
        "Answer with the JSON object only.",
    ]
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


# --- The Party Actor -------------------------------------------------------------------------------


@dataclass
class LanguageModelActor:
    """A Party Actor that asks a language model, behind the same interface as the Stand-in."""

    client: ChatClient
    config: ModelConfig
    cache: ResponseCache
    masker: Masker
    horizon: int
    seed: int = 0
    usage: UsageLedger = field(default=None)
    name: str = "language-model"

    def __post_init__(self):
        if self.usage is None:
            self.usage = UsageLedger(self.config.input_price, self.config.output_price)

    def for_run(self, run: int) -> LanguageModelActor:
        """The same Actor for one Run under per-Run decisions: its own seed, the shared ledger."""
        return LanguageModelActor(
            self.client, self.config, self.cache, self.masker, self.horizon, self.seed + run, self.usage, self.name
        )

    def decide(self, state: StateSummary, menu: tuple[Move, ...] = MENU) -> Decision:
        messages = masked_prompt(state, self.masker, self.horizon, menu)
        for attempt in range(2):
            text = self._ask(messages)
            decision = _parse(text, menu)
            if decision is not None:
                if attempt:
                    self.usage.retries += 1
                return decision
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not a valid answer. Reply with the JSON object only, "
                 f"with a move written exactly as one of: {', '.join(map(str, menu))}."},
            ]
        self.usage.retries += 1
        self.usage.invalid_answers += 1
        return Decision(Move("hold"), "invalid answer")

    def _ask(self, messages: list[dict]) -> str:
        key = ResponseCache.key(messages, self.seed, self.config)
        cached = self.cache.get(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached.text
        response = self.client.complete(messages, self.seed)
        self.cache.put(key, response)
        self.usage.calls += 1
        self.usage.prompt_tokens += response.usage.prompt_tokens
        self.usage.completion_tokens += response.usage.completion_tokens
        return response.text


def _parse(text: str, menu: tuple[Move, ...]) -> Decision | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        answer = json.loads(text[start : end + 1])
        move = Move.parse(str(answer["move"]))
    except (ValueError, KeyError, TypeError):
        return None
    if move not in menu:
        return None
    return Decision(move, str(answer.get("reason", "")).strip() or "no reason given")


# --- Cutoff Labels and the leak flag ------------------------------------------------------------


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def cutoff_label(first: date, last: date, training_cutoff: date, margin_months: int = 3) -> str:
    """Before: the whole window ends before the cutoff. After: it starts more than the safety
    margin past it, because published cutoffs are fuzzy. Straddling: anything in between."""
    if last < training_cutoff:
        return "Before"
    if first > _add_months(training_cutoff, margin_months):
        return "After"
    return "Straddling"


def hindsight_leak_flag(llm: float, stand_in: float, statistical: float) -> bool:
    """A possible Hindsight Leak: only the language-model Twin beats the statistical Baseline (CRPS)."""
    return llm < statistical <= stand_in

