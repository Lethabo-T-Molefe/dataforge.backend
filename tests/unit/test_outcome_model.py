"""The generic outcome-model engine: column plan, data checks, validation against baselines, fairness, drift."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dataforge_backend.domain import outcome_model as om
from dataforge_backend.domain.prediction import ModelError
from dataforge_backend.seed_data import learners as showcase

PROTECTED = ["gender", "population_group", "disability", "age_band"]


def _fields(frame: pd.DataFrame, pii: set[str]) -> list[dict[str, Any]]:
    """The shape the catalogue caches for a silver table (see application.catalogue.compute_fields)."""
    out = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_integer_dtype(series):
            kind = "integer"
        elif pd.api.types.is_float_dtype(series):
            kind = "numeric"
        else:
            kind = "text"
        values = [str(v) for v in series.dropna().astype(str).value_counts().index[:200]]
        out.append(
            {
                "name": name,
                "type": kind,
                "distinct": int(series.nunique(dropna=True)),
                "pii": name in pii,
                "values": values if kind == "text" else [],
                "monthLike": kind == "text"
                and bool(values)
                and all(len(v) == 7 and v[4] == "-" for v in values),
            }
        )
    return out


@lru_cache(maxsize=1)
def _learners() -> pd.DataFrame:
    frame = pd.DataFrame(showcase.learners(2000)).drop(columns=["id_number"])
    for column in ("formative_avg", "attendance_pct", "travel_km"):
        frame[column] = frame[column].astype(float)
    return frame


def _learner_spec(**overrides: Any) -> om.TrainingSpec:
    body = {
        "name": "learner-dropout-risk",
        "version": "1.0",
        "outcome": {
            "column": "outcome",
            "positive": "Dropped out",
            "unlabelled": ["In progress"],
            "label": "drops out",
            "entity": "learner",
        },
        "protected": PROTECTED,
        "trainUntil": "2025-04",
        "cvFolds": 3,
        **overrides,
    }
    return om.spec_from_dict(body)


@lru_cache(maxsize=1)
def _trained() -> tuple[om.TrainingSpec, list[om.ColumnPlan], dict[str, Any]]:
    frame = _learners()
    spec = _learner_spec()
    plan = om.suggest_columns(_fields(frame, {"full_name", "cell_number"}), len(frame), spec)
    return spec, plan, om.train(frame, spec, plan)


def _roles(plan: list[dict[str, Any]]) -> dict[str, str]:
    return {p["name"]: p["role"] for p in plan}


# ------------------------------------------------------------------ specification and plan


def test_a_training_request_is_checked_with_every_problem_named() -> None:
    with pytest.raises(ModelError) as error:
        om.spec_from_dict({"name": "Bad Name", "outcome": {}, "threshold": {"policy": "recall_at"}})
    message = str(error.value)
    for expected in ("name", "version", "outcome", "value that counts", "between 0 and 1"):
        assert expected in message
    spec = _learner_spec(threshold={"policy": "top_pct", "target": 0.2})
    assert spec.threshold_policy == "top_pct" and spec.threshold_target == 0.2
    assert spec.outcome.unlabelled == ("In progress",)


def test_the_plan_keeps_personal_and_protected_columns_out_of_the_features() -> None:
    frame = _learners()
    plan = om.suggest_columns(_fields(frame, {"full_name", "cell_number"}), len(frame), _learner_spec())
    roles = {p.name: p.role for p in plan}
    assert roles["outcome"] == "target"
    assert roles["full_name"] == roles["cell_number"] == "excluded"
    assert all(roles[name] == "protected" for name in PROTECTED)
    assert roles["learner_id"] == "key"
    assert roles["start_month"] == "time"
    assert roles["starts"] == "excluded"  # the same for every learner
    assert roles["attendance_pct"] == roles["employer"] == "feature"
    reasons = {p.name: p.reason for p in plan}
    assert "never used to score" in reasons["gender"]
    assert "Personal information" in reasons["full_name"]


def test_protected_attributes_are_suggested_from_their_names() -> None:
    suggested = om.suggest_protected(_fields(_learners(), {"full_name"}))
    assert set(PROTECTED) <= set(suggested)
    assert "attendance_pct" not in suggested and "stipend_rand_avg" not in suggested


def test_labels_leave_unknown_outcomes_out_of_learning() -> None:
    frame = pd.DataFrame({"outcome": ["Dropped out", "Completed", "In progress", "", None, "dropped out"]})
    labelled, y = om.labels(frame, om.OutcomeSpec("outcome", "Dropped out", ("In progress",)))
    assert labelled.tolist() == [True, True, False, False, False, True]
    assert y[labelled].tolist() == [1, 0, 1]
    numeric = pd.DataFrame({"churned": [1, 0, 1.0, "1"]})
    _, y2 = om.labels(numeric, om.OutcomeSpec("churned", "1"))
    assert y2.tolist() == [1, 0, 1, 1]


# ------------------------------------------------------------------ training on the showcase


def test_leaky_columns_are_found_and_left_out() -> None:
    _, _, result = _trained()
    plan = {p["name"]: p for p in result["plan"]}
    # completed and dropped are recorded after the outcome: each alone predicts it almost perfectly.
    for column in ("completed", "dropped"):
        assert plan[column]["role"] == "excluded", plan[column]
        assert "leakage" in plan[column]["reason"]
    assert result["data"]["labelled"] >= om.MIN_LABELLED
    assert 0.18 <= result["data"]["prevalence"] <= 0.35
    assert "2025-04" in result["data"]["notes"][0]


def test_the_trained_models_validate_and_beat_the_documented_heuristic() -> None:
    _, _, result = _trained()
    heuristic = result["baselines"]["heuristic"]["rocAuc"]
    assert {m["algorithm"] for m in result["models"]} == {"logistic", "hist_gb"}
    for model in result["models"]:
        metrics = model["metrics"]
        assert metrics["rocAuc"] > 0.7, metrics["rocAuc"]
        assert 0 <= metrics["brier"] < 0.25
        assert metrics["crossValidation"]["folds"] == 3
        assert 0 < model["threshold"] < 1
        assert metrics["atThreshold"]["tp"] + metrics["atThreshold"]["fn"] > 0
        assert len(metrics["roc"]) <= 50 and metrics["calibration"]
    best = max(m["metrics"]["rocAuc"] for m in result["models"])
    assert best > heuristic, (best, heuristic)
    assert result["baselines"]["prevalence"]["rocAuc"] == 0.5
    assert result["baselines"]["bestSingleColumn"]["column"] not in ("completed", "dropped")


def test_the_fairness_audit_covers_every_protected_attribute() -> None:
    _, _, result = _trained()
    fairness = result["models"][0]["fairness"]
    assert fairness["audited"] is True
    assert {a["attribute"] for a in fairness["attributes"]} == set(PROTECTED)
    disability = next(a for a in fairness["attributes"] if a["attribute"] == "disability")
    # About 3% of learners report a disability: too few in the test split to judge either way.
    assert "Yes" in disability["tooSmallGroups"]
    for attribute in fairness["attributes"]:
        for group in attribute["groups"]:
            assert 0 <= group["selectionRate"] <= 1


def test_a_stored_model_scores_and_explains_records_and_refuses_if_tampered() -> None:
    spec, _, result = _trained()
    model = result["models"][0]
    meta = {
        "sha256": model["artifactSha256"],
        "runtime": result["runtime"],
        "threshold": model["threshold"],
        "outcome": {
            "column": spec.outcome.column,
            "positive": spec.outcome.positive,
            "unlabelled": list(spec.outcome.unlabelled),
            "label": spec.outcome.label,
            "entity": spec.outcome.entity,
        },
    }
    current = _learners()[_learners()["outcome"] == "In progress"].head(40)
    scored = om.score(current, model["artifact"], meta)
    assert len(scored["probability"]) == 40
    assert all(0 <= p <= 1 for p in scored["probability"])
    assert set(scored["band"]) <= {"High", "Watch", "Low"}
    assert all(len(d) <= 3 for d in scored["drivers"])
    first = next(d for d in scored["drivers"] if d)
    assert "likelihood that the learner drops out" in first[0]["text"]
    with pytest.raises(om.ArtifactError):
        om.load_artifact(model["artifact"], "0" * 64, result["runtime"])
    with pytest.raises(om.ArtifactError):
        om.load_artifact(model["artifact"], model["artifactSha256"], {"sklearn": "0.1.0"})


def test_logistic_contributions_add_up_to_the_models_log_odds() -> None:
    spec, _, result = _trained()
    model_bytes = next(m for m in result["models"] if m["algorithm"] == "logistic")
    model = om.load_artifact(model_bytes["artifact"], model_bytes["artifactSha256"], result["runtime"])
    X = om._feature_frame(_learners().head(25), model["numeric"], model["categorical"])
    columns = model["numeric"] + model["categorical"]
    drivers = om.explain(model, X, spec.outcome, top=len(columns))
    logit = model["pipeline"].decision_function(X)
    intercept = float(model["pipeline"].named_steps["model"].intercept_[0])
    for i, row in enumerate(drivers):
        assert sum(d["effect"] for d in row) + intercept == pytest.approx(logit[i], abs=0.01)


def test_drift_is_measured_against_the_training_distributions() -> None:
    _, _, result = _trained()
    frame = _learners()
    early = frame[frame["start_month"] <= "2025-04"]
    same = om.psi(result["reference"], early)
    assert same["status"] == "stable", same["columns"][:3]
    moved = early.assign(lms_logins=early["lms_logins"] * 2.5)
    drifted = om.psi(result["reference"], moved)
    worst = drifted["columns"][0]
    assert worst["column"] == "lms_logins" and worst["psi"] >= om.PSI_DRIFT and drifted["status"] == "drift"


def test_the_model_card_states_use_limits_and_evidence() -> None:
    spec, _, result = _trained()
    card = om.model_card(
        spec, result, result["models"][0], om.CardContext("Learner progress", 1, "Zola Nkosi", "2026-11-19")
    )
    assert "drops out" in card["predicts"]
    assert any("section 71" in item for item in card["notFor"])
    assert card["performance"]["liftOverHeuristic"] is not None
    assert all(f["role"] == "feature" for f in card["features"])
    assert any(e["name"] == "gender" for e in card["excluded"])


# ------------------------------------------------------------------ the engine is not tied to learners


def test_any_dataset_with_a_yes_no_outcome_can_be_modelled() -> None:
    rng = np.random.default_rng(7)
    n = 1500
    tenure = rng.integers(1, 72, n)
    calls = rng.poisson(2, n)
    spend = rng.normal(450, 120, n).round(2)
    plan_type = rng.choice(["Basic", "Plus", "Premium"], n, p=[0.5, 0.3, 0.2])
    logit = -0.4 - 0.05 * tenure + 0.6 * calls + np.where(plan_type == "Basic", 0.6, 0.0)
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "support_calls": calls,
            "monthly_spend": spend,
            "plan": plan_type,
            "churned": churned,
        }
    )
    spec = om.spec_from_dict(
        {"name": "churn", "version": "1.0", "outcome": {"column": "churned", "positive": "Yes"}, "cvFolds": 3}
    )
    plan = om.suggest_columns(_fields(frame, set()), n, spec)
    assert _roles([p.to_dict() for p in plan])["customer_id"] == "key"
    result = om.train(frame, spec, plan)
    assert all(m["metrics"]["rocAuc"] > 0.7 for m in result["models"])
    assert result["models"][0]["fairness"]["audited"] is False


def test_too_little_labelled_data_is_refused_with_the_reason() -> None:
    frame = pd.DataFrame({"x": range(60), "y": ["a"] * 55 + ["b"] * 5})
    spec = om.spec_from_dict({"name": "tiny", "version": "1.0", "outcome": {"column": "y", "positive": "b"}})
    plan = om.suggest_columns(_fields(frame, set()), len(frame), spec)
    with pytest.raises(ModelError, match="at least"):
        om.train(frame, spec, plan)


def test_fairness_arithmetic_on_a_small_example() -> None:
    y = np.array([1, 1, 0, 0] * 20 + [1, 0, 0, 0] * 20)
    flagged = np.array([True, True, False, False] * 20 + [False, False, False, False] * 20)
    groups = {"group": pd.Series(["A"] * 80 + ["B"] * 80)}
    report = om.fairness_report(y, flagged, groups)
    attribute = report["attributes"][0]
    a, b = attribute["groups"]
    assert (a["recall"], b["recall"]) == (1.0, 0.0)
    assert attribute["recallGap"] == 1.0 and attribute["selectionRateRatio"] == 0.0
    assert report["flagged"] is True and len(attribute["flags"]) == 2
