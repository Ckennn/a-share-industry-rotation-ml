"""Validated experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates the public experiment contract."""


@dataclass(frozen=True)
class SplitSpec:
    train_end: str
    validation_end: str
    test_end: str

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end < self.test_end:
            raise ConfigError("split dates must satisfy train < validation < test")

    def label(self, decision_date: str) -> str:
        if decision_date <= self.train_end:
            return "train"
        if decision_date <= self.validation_end:
            return "validation"
        return "test"


@dataclass(frozen=True)
class PortfolioSpec:
    top_n: int = 3
    cost_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ConfigError("top_n must be positive")
        if self.cost_rate < 0:
            raise ConfigError("cost_rate cannot be negative")


@dataclass(frozen=True)
class BootstrapSpec:
    block_months: int = 6
    simulations: int = 2000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.block_months < 1 or self.simulations < 1:
            raise ConfigError("bootstrap block and simulations must be positive")


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class ExperimentConfig:
    split: SplitSpec
    portfolio: PortfolioSpec
    bootstrap: BootstrapSpec
    models: Mapping[str, tuple[ModelCandidate, ...]]
    walk_forward_start: str
    walk_forward_end: str
    min_core_train_months: int
    validation_months: int
    random_top3_simulations: int = 1000
    random_state: int = 42
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.walk_forward_start > self.walk_forward_end:
            raise ConfigError("walk-forward start must not exceed end")
        if self.min_core_train_months < 1 or self.validation_months < 1:
            raise ConfigError("training and validation month counts must be positive")
        if self.random_top3_simulations < 1:
            raise ConfigError("random_top3_simulations must be positive")
        if not self.models or any(not candidates for candidates in self.models.values()):
            raise ConfigError("every configured model must contain at least one candidate")


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"unknown {context} keys: {sorted(unknown)}")


def load_config(path: Path) -> ExperimentConfig:
    """Load YAML and reject keys outside the versioned public schema."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    top_keys = {
        "split",
        "portfolio",
        "bootstrap",
        "models",
        "walk_forward_start",
        "walk_forward_end",
        "min_core_train_months",
        "validation_months",
        "random_state",
        "n_jobs",
        "random_top3_simulations",
    }
    _reject_unknown(raw, top_keys, "top-level")

    split_raw = raw.get("split")
    portfolio_raw = raw.get("portfolio")
    bootstrap_raw = raw.get("bootstrap")
    model_raw = raw.get("models")
    if not all(isinstance(value, dict) for value in (split_raw, portfolio_raw, bootstrap_raw, model_raw)):
        raise ConfigError("split, portfolio, bootstrap, and models must be mappings")
    _reject_unknown(split_raw, {"train_end", "validation_end", "test_end"}, "split")
    _reject_unknown(portfolio_raw, {"top_n", "cost_rate"}, "portfolio")
    _reject_unknown(bootstrap_raw, {"block_months", "simulations", "seed"}, "bootstrap")

    models: dict[str, tuple[ModelCandidate, ...]] = {}
    for model_name, candidates in model_raw.items():
        if not isinstance(candidates, list):
            raise ConfigError(f"model {model_name} candidates must be a list")
        parsed: list[ModelCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ConfigError(f"model {model_name} candidate must be a mapping")
            _reject_unknown(candidate, {"id", "params"}, f"{model_name} candidate")
            if not isinstance(candidate.get("params"), dict):
                raise ConfigError(f"model {model_name} params must be a mapping")
            parsed.append(ModelCandidate(str(candidate["id"]), dict(candidate["params"])))
        models[str(model_name)] = tuple(parsed)

    required = {
        "walk_forward_start",
        "walk_forward_end",
        "min_core_train_months",
        "validation_months",
    }
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"missing required configuration keys: {sorted(missing)}")
    return ExperimentConfig(
        split=SplitSpec(**split_raw),
        portfolio=PortfolioSpec(**portfolio_raw),
        bootstrap=BootstrapSpec(**bootstrap_raw),
        models=models,
        walk_forward_start=str(raw["walk_forward_start"]),
        walk_forward_end=str(raw["walk_forward_end"]),
        min_core_train_months=int(raw["min_core_train_months"]),
        validation_months=int(raw["validation_months"]),
        random_top3_simulations=int(raw.get("random_top3_simulations", 1000)),
        random_state=int(raw.get("random_state", 42)),
        n_jobs=int(raw.get("n_jobs", 1)),
    )
