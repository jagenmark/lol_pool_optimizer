from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    parameter: float | None = None


def build_training_frame(
    summary_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach champion-level priors to patch-A matchup rows."""
    overall_rates = summary_df[["champion_id", "overall_winrate"]].copy()
    training_df = matchup_df.merge(overall_rates, on="champion_id", how="left")
    if training_df["overall_winrate"].isna().any():
        raise ValueError("Some matchup rows are missing champion overall win rates")
    return training_df


def estimate_matchup_winrates(training_df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Estimate matchup win rates for one model specification."""
    estimated = training_df.copy()
    observed = estimated["matchup_winrate"].to_numpy(dtype=float)

    if spec.model_name == "raw":
        estimated["estimated_winrate"] = observed
        return estimated

    if spec.model_name == "shrink_overall":
        if spec.parameter is None:
            raise ValueError("shrink_overall requires a shrinkage parameter c")
        games = estimated["matchup_games"].to_numpy(dtype=float)
        priors = estimated["overall_winrate"].to_numpy(dtype=float)
        shrink = games / (games + float(spec.parameter))
        estimated["estimated_winrate"] = shrink * observed + (1.0 - shrink) * priors
        estimated["shrinkage_weight"] = shrink
        return estimated

    raise ValueError(f"Unsupported model: {spec.model_name}")


def build_model_grid(shrinkage_c_values: tuple[float, ...]) -> list[ModelSpec]:
    """Return the first experiment's model grid."""
    models = [ModelSpec(model_name="raw", parameter=None)]
    models.extend(
        ModelSpec(model_name="shrink_overall", parameter=float(c_value))
        for c_value in shrinkage_c_values
    )
    return models


def format_model_parameter(spec: ModelSpec) -> str:
    if spec.parameter is None:
        return "none"
    if float(spec.parameter).is_integer():
        return str(int(spec.parameter))
    return str(spec.parameter)


def empirical_bayes_extension_placeholder() -> None:
    """
    Placeholder for a later beta-binomial / empirical Bayes implementation.

    The current pipeline routes all matchup estimates through `estimate_matchup_winrates`,
    so a future EB model can be added here without rewriting the loader, optimizer,
    or evaluation code.
    """
