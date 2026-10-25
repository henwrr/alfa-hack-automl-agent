"""Схемы ответа LLM (pydantic v2).

ExperimentSpec — один эксперимент: модель, параметры, таргет/веса, признаки.
Proposal — ответ исследователя целиком. Его JSON Schema уходит в structured output
(Ollama `format`, OpenAI-совместимый `response_format`), чтобы модель не возвращала битый JSON.

Эксперименты внутри Proposal проверяются по одному (в runner), чтобы одна плохая
спецификация не выбрасывала весь ответ.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .features import OPS_DOC, TARGET_OPS_DOC

Family = Literal["lightgbm", "catboost", "xgboost", "extratrees", "linear"]
OpName = Literal[tuple({**OPS_DOC, **TARGET_OPS_DOC})]  # type: ignore[valid-type]

# Это задаёт хост: потоки, seed, логирование, число деревьев (его выбирает early stopping).
BANNED_PARAMS = {"n_jobs", "thread_count", "nthread", "num_threads", "device", "device_type", "task_type",
                 "gpu_id", "random_state", "random_seed", "seed", "verbose", "verbosity", "callbacks",
                 "train_dir", "allow_writing_files", "early_stopping_rounds", "eval_metric", "cat_features",
                 "n_estimators", "iterations", "num_boost_round", "num_iterations"}


class FeatureOp(BaseModel):
    """Операция признаков; остальные поля зависят от op (cols, pairs, by, code, ...)."""
    model_config = ConfigDict(extra="allow")
    op: OpName


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    family: Family = "lightgbm"
    params: dict[str, int | float | str | bool] = Field(default_factory=dict)
    target_transform: Literal["none", "log1p", "sqrt", "gamma"] = "none"
    offset: Literal["none", "lag"] = "none"
    class_weight: Literal["none", "balanced", "sqrt_balanced"] = "none"
    early_stopping_rounds: int | None = Field(None, ge=20, le=1000,
                                              description="терпение early stopping; по умолчанию 100")
    features: list[FeatureOp] = Field(default_factory=list)

    @field_validator("params", mode="before")
    @classmethod
    def _clean_params(cls, value):
        if not isinstance(value, dict):
            return {}
        out = {k: v for k, v in value.items() if k not in BANNED_PARAMS and isinstance(v, (int, float, str, bool))}
        if "learning_rate" in out:
            out["learning_rate"] = float(min(max(float(out["learning_rate"]), 0.01), 0.3))
        return out

    @field_validator("features")
    @classmethod
    def _dedupe(cls, ops):
        seen, out = set(), []
        for op in ops:
            key = json.dumps(op.model_dump(), sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                out.append(op)
        return out


class ExperimentDraft(BaseModel):
    """Элемент списка экспериментов в ответе LLM (spec проверяется отдельно)."""
    model_config = ConfigDict(extra="ignore")
    hypothesis: str = ""
    expected: str = ""
    spec: dict = Field(default_factory=dict)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    analysis: str = ""
    experiments: list[ExperimentDraft] = Field(default_factory=list)
    stop: bool = False


def proposal_json_schema() -> dict:
    """Схема для structured output: spec описан полностью (+ extends_best/remove_features)."""
    schema = Proposal.model_json_schema()
    spec_schema = ExperimentSpec.model_json_schema()
    spec_schema.setdefault("properties", {}).update({
        "extends_best": {"type": "boolean"},
        "remove_features": {"type": "array", "items": {"type": "object"}},
    })
    defs = {**schema.pop("$defs", {}), **spec_schema.pop("$defs", {})}
    defs["ExperimentSpec"] = spec_schema
    defs["ExperimentDraft"]["properties"]["spec"] = {"$ref": "#/$defs/ExperimentSpec"}
    schema["$defs"] = defs
    return schema


def format_validation_error(exc: ValidationError) -> str:
    """Короткое понятное сообщение для LLM: где ошибка и что ожидалось."""
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
