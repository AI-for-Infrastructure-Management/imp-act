"""
Create budget scenarios for a preset using a simple heuristic policy and
median per-period maintenance costs, inspired by evaluation/get_budget.py,
but without depending on jaxmarl. The rollouts are collected with the IMP‑act
JAX environment directly.

Steps:
1) Load an unconstrained preset ("<preset>-unconstrained-jax").
2) Run a heuristic policy for N rollouts, collect per-step maintenance costs.
3) Aggregate to per-period costs and take distribution quantiles to derive
   budget amounts.
4) Optional: write variant YAMLs that include the base preset.

Usage examples (Hydra)
- Use bundled config (dry run by default in YAML):
  python create_budget_scenarios.py

- Override preset and run dry-run explicitly:
  python create_budget_scenarios.py preset=Cologne-v1 dry_run=true

- Compute and write variants (disable dry-run, allow overwrite):
  python create_budget_scenarios.py preset=6.7_7.2_50.5_51 dry_run=false overwrite=true

- Point to a different policy file:
  python create_budget_scenarios.py policy=imp-act/imp_act/environments/dev/humble_heuristic.yaml
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import numpy as np
import yaml
import logging

# Silence JAX backend probing logs (do not force CPU/GPU selection)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig

from imp_act import make


SCRIPT_DIR = Path(__file__).resolve().parent


def round_to_nearest_multiple(x: float, base: int = 5) -> float:
    """Round x to a human-friendly multiple.

    - Handles non-finite inputs (NaN/inf) and zeros gracefully.
    - Prevents overflow for tiny magnitudes by clamping the scale.
    """
    if not np.isfinite(x):
        return 0.0
    if x == 0:
        return 0.0
    ax = abs(float(x))
    # Determine order of magnitude; clamp to non-negative for factor
    scale = int(np.floor(np.log10(ax)))
    factor = 10 ** max(scale - 1, 0)
    return round(x / (base * factor)) * (base * factor)


def human_readable_scale(x: float) -> Tuple[float, str]:
    ax = abs(x)
    if ax >= 1e12:
        return 1e12, "T"
    if ax >= 1e9:
        return 1e9, "B"
    if ax >= 1e6:
        return 1e6, "M"
    if ax >= 1e3:
        return 1e3, "K"
    return 1.0, ""


def _check_cfg(cfg: DictConfig) -> None:
    """Minimal validation for required fields and shapes."""
    required = [
        "preset",
        "seed",
        "period",
        "quantiles",
        "labels",
        "dry_run",
        "overwrite",
        "policy_parameters",
        "num_rollouts",
    ]
    missing = [k for k in required if not hasattr(cfg, k)]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    seed = int(cfg.seed)
    period = int(cfg.period)
    num_rollouts = int(cfg.num_rollouts)
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if period <= 0:
        raise ValueError("period must be > 0")
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be > 0")

    qs = [float(x) for x in list(cfg.quantiles)]
    if not qs:
        raise ValueError("quantiles must be a non-empty list of floats")
    lbls = [str(x) for x in list(cfg.labels)]
    if not lbls:
        raise ValueError("labels must be a non-empty list of strings")
    if len(lbls) != len(qs):
        raise ValueError("labels length must match quantiles length")


# ------------------------------- policy -------------------------------------
def _load_policy_params(params: dict) -> dict:
    """Validate and coerce inline policy parameters.

    Expects a mapping with keys:
    - inspection_interval: int (> 0)
    - repair_threshold: int (>= 0)
    """
    if not isinstance(params, dict):
        raise ValueError("policy_parameters must be a mapping")

    if "inspection_interval" not in params or "repair_threshold" not in params:
        raise ValueError(
            "policy_parameters must include 'inspection_interval' and 'repair_threshold'"
        )

    interval = int(params["inspection_interval"])
    threshold = int(params["repair_threshold"])
    if interval <= 0:
        raise ValueError("inspection_interval must be > 0")
    if threshold < 0:
        raise ValueError("repair_threshold must be >= 0")

    print(
        f"Using policy parameters: inspection_interval={interval}, repair_threshold={threshold}"
    )
    return {"inspection_interval": interval, "repair_threshold": threshold}


def _policy_from_params(params: dict):
    """
    Return a policy function that uses `inspection_interval` and `repair_threshold`.
    This "factory" returns a closure capturing those parameters.

    Parameters (must be provided)
    - inspection_interval: int
    - repair_threshold: int
    """
    interval = int(params["inspection_interval"])
    threshold = int(params["repair_threshold"])

    def policy(key, state, obs, env):  # jitted-friendly signature
        # Support both wrapped and bare JAX env states
        s = getattr(state, "env_state", state)
        tstep = s.timestep
        obs_insp = s.observation

        # Step 1: default action = 0
        actions = jnp.zeros_like(obs_insp, dtype=jnp.int32)

        # Step 2: if tstep % interval == 0 => action 1
        actions = jnp.where(tstep % interval == 0, 1, actions)

        # Step 3: if obs_insp >= threshold => action 2 (priority)
        actions = jnp.where(obs_insp >= threshold, 2, actions)
        return actions

    return policy


# ------------------------------ rollouts ------------------------------------
def generate_per_period_costs(
    env,
    policy_fn,
    rollouts: int,
    horizon: int,
    period: int,
    seed: int = 88,
) -> np.ndarray:
    """Generate rollout data and return per-period maintenance costs.

    Outline
    - 1) Seed -> batch of per-rollout keys
    - 2) Per-rollout function: reset once, then scan for a fixed horizon
         collecting maintenance rewards at each step
    - 3) Vectorize with vmap and JIT for speed
    - 4) Aggregate step-wise rewards into per-period totals
    Returns per_period_costs (rollouts, periods).
    """

    # 1) Derive a base PRNGKey from seed and split one key per rollout
    key = jax.random.PRNGKey(seed)
    keys_batch = jax.random.split(key, rollouts)

    def run_rollout(key):
        # 2) Rollout: reset once, then scan over steps
        key, key_reset = jax.random.split(key)
        obs, state = env.reset(key_reset)

        def scan_step(carry, _):
            # 2.a) Unpack scan carry (track done to split episodes cleanly)
            key, last_obs, last_state, done_prev = carry

            # 2.b) Policy action
            key, key_act = jax.random.split(key)
            actions = policy_fn(key_act, last_state, last_obs, env)

            # 2.c) Environment step
            key, key_step = jax.random.split(key)
            obs, state, reward, done, info = env.step(key_step, last_state, actions)
            maintenance_reward = info["reward_elements"]["maintenance_reward"]

            # 2.d) Mask after first episode done; carry forward 'done' flag
            maintenance_reward = jnp.where(done_prev, 0.0, maintenance_reward)
            done_out = jnp.logical_or(done_prev, done)
            return (key, obs, state, done_out), maintenance_reward

        # 2.e) Reset once per rollout
        key, key_reset = jax.random.split(key)
        obs, state = env.reset(key_reset)

        # 2.f) Separate scan RNG for clarity
        key, key_scan = jax.random.split(key)
        carry, maintenance_rewards = jax.lax.scan(
            scan_step,
            (key_scan, obs, state, False),
            None,
            length=horizon,
        )
        return maintenance_rewards

    # 3) Vectorize rollouts (vmap) and JIT compile for speed
    batched = jax.jit(jax.vmap(run_rollout, in_axes=(0,)))
    maintenance_rewards_bt = batched(keys_batch)  # (rollouts, T)

    # 4) Aggregate step-wise rewards into per-period totals
    def agg_period(maintenance_rewards_row):
        """Aggregate step-wise rewards into period totals for one rollout.

        - Compute how many full periods fit in the horizon (nP)
        - Trim any trailing steps that don't complete a full period
        - Reshape to (nP, period) and sum within each period
        Returns a vector of length nP with per-period sums.
        """
        T = maintenance_rewards_row.shape[0]
        nP = T // period  # number of complete periods
        trimmed = maintenance_rewards_row[: nP * period]
        return trimmed.reshape(nP, period).sum(axis=1)

    per_period_costs = jax.vmap(agg_period)(maintenance_rewards_bt)
    return np.array(per_period_costs)


# ------------------------------- IO helpers ---------------------------------
def _resolve_preset_dir(preset: str) -> Path:
    """Resolve preset directory with minimal, deterministic logic.

    Order tried:
    1) `preset` as given (absolute or relative path).
    2) `<script_dir>/../presets/<preset>`.
    """
    p = Path(preset)
    if p.is_dir():
        return p

    candidate = (SCRIPT_DIR.parent / "presets" / preset).resolve()
    if candidate.is_dir():
        return candidate

    raise ValueError(f"Preset directory not found: {p} | {candidate}")


def _dump_yaml_with_spacing(data: dict) -> str:
    """Dump YAML with blank lines between core sections; keep a list inline."""
    class _Flow(list):
        pass
    yaml.SafeDumper.add_representer(
        _Flow,
        lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", list(v), flow_style=True),
    )

    # Inline maintenance.initial_damage_distribution if present
    if isinstance(data, dict) and isinstance(data.get("maintenance"), dict):
        ild = data["maintenance"].get("initial_damage_distribution")
        if isinstance(ild, list):
            data = {**data, "maintenance": {**data["maintenance"], "initial_damage_distribution": _Flow(ild)}}

    order = ["include", "maintenance", "traffic", "topology"]
    pieces = [
        yaml.dump({k: data[k]}, Dumper=yaml.SafeDumper, sort_keys=False).strip()
        for k in order if k in data
    ]
    pieces += [
        yaml.dump({k: v}, Dumper=yaml.SafeDumper, sort_keys=False).strip()
        for k, v in data.items() if k not in order
    ]
    return ("\n\n".join(pieces) + "\n") if pieces else yaml.safe_dump(data, sort_keys=False)


def _write_variant_yaml(
    base_dir: Path, label: str, base_name: str, budget: float
) -> Path:
    # Only include sections that carry meaning for the variant.
    # For budget variants, traffic/topology are unchanged — omit empty sections.
    data = {
        "include": {"path": f"./{base_name}.yaml", "override": False},
        "maintenance": {
            "enforce_budget_constraint": True,
            "budget_amount": float(budget),
        },
    }
    # Keep section ordering and blank lines for readability
    out_path = base_dir / f"{base_name}-{label}.yaml"
    text = _dump_yaml_with_spacing(data)
    out_path.write_text(text)
    return out_path


def _update_base_budget(base_yaml: Path, budget: float) -> None:
    """Overwrite the base preset YAML with the limited-budget settings.

    Sets maintenance.enforce_budget_constraint=true and maintenance.budget_amount=budget.
    Keeps other sections untouched and inserts a blank line before the maintenance header.
    """
    data = yaml.safe_load(base_yaml.read_text()) or {}
    maint = data.setdefault("maintenance", {})
    maint["enforce_budget_constraint"] = True
    maint["budget_amount"] = float(budget)

    text = _dump_yaml_with_spacing(data)
    base_yaml.write_text(text)


@hydra.main(
    config_path=".", config_name="create_budget_scenarios_config", version_base=None
)
def main(cfg: DictConfig) -> None:
    # Hydra can reconfigure logging to INFO; re‑silence JAX probing logs
    import logging as _logging
    _logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
    _logging.getLogger("jax._src.xla_bridge").propagate = False
    _check_cfg(cfg)

    # Resolve preset directory (name or path)
    preset = cfg.preset
    preset_dir = _resolve_preset_dir(str(preset))
    if not preset_dir.exists() or not preset_dir.is_dir():
        raise ValueError(f"Preset directory not found: {preset_dir}")
    base_yaml = preset_dir / f"{preset_dir.name}.yaml"
    if not base_yaml.exists():
        raise ValueError(f"Base preset YAML not found: {base_yaml}")

    # Build unconstrained JAX env variant
    env = make(f"{preset_dir.name}-unconstrained-jax")
    env.budget_amount = 0.0  # ensure unconstrained
    horizon = (
        int(env.env.max_timesteps) if hasattr(env, "env") else int(env.max_timesteps)
    )

    # Optionally override horizon from config (falls back to env default)
    if hasattr(cfg, "num_steps") and cfg.num_steps is not None:
        horizon = int(cfg.num_steps)

    # Policy: load inline from cfg.policy_parameters
    params = _load_policy_params(dict(cfg.policy_parameters))
    policy_fn = _policy_from_params(params)

    # Rollouts via the API
    PERIOD = int(cfg.period)
    rollouts_val = int(cfg.num_rollouts)

    per_period_costs = generate_per_period_costs(
        env=env,
        policy_fn=policy_fn,
        rollouts=rollouts_val,
        horizon=horizon,
        period=PERIOD,
        seed=int(cfg.seed),
    )
    # Mean across rollouts gives the typical cost per period
    cycle_costs = per_period_costs.mean(axis=0)  # (periods,)

    qs = [float(x) for x in cfg.quantiles]
    labels = [str(x) for x in cfg.labels]
    if len(labels) != len(qs):
        raise ValueError("labels must match the number of quantiles")

    qvals = np.quantile(cycle_costs, qs)
    # Convert negative costs to positive budget magnitudes and round nicely
    budgets_pos = [-float(v) for v in qvals]
    budgets_rounded = [round_to_nearest_multiple(b) for b in budgets_pos]

    # Print summary with tidy alignment
    norm, unit = human_readable_scale(np.median(np.abs(qvals)))
    label_w = max(len(l) for l in labels) if labels else 0
    for q, lab, raw, rnd in zip(qs, labels, qvals, budgets_rounded):
        pct = int(round(q * 100))
        budget_val = -raw / norm
        budget_rnd = rnd / norm
        print(
            f"{lab:<{label_w}} ({pct:>3}th percentile) | "
            f"budget: {budget_val:>7.2f}{unit}  (~ {budget_rnd:>7.2f}{unit})"
        )

    if bool(cfg.dry_run):
        return

    # Write variants
    written = []
    for lab, b in zip(labels, budgets_rounded):
        # Treat 'limited-budget' as the default: update the base preset directly
        if lab == "limited-budget":
            _update_base_budget(base_yaml, b)
            print(
                f"Updated base preset '{base_yaml.name}' with limited-budget amount: {b:.0f}"
            )
            continue
        out = preset_dir / f"{preset_dir.name}-{lab}.yaml"
        if out.exists() and not bool(cfg.overwrite):
            print(f"Skip existing (use overwrite=true): {out}")
            continue
        written.append(_write_variant_yaml(preset_dir, lab, preset_dir.name, b))
    if written:
        print("Wrote:")
        for pth in written:
            print(f"- {pth}")


if __name__ == "__main__":
    main()
