"""
Compute and set the travel_time_reward_factor for a preset.

Usage examples
- Compute and update the base config:
  python compute_reward_factor.py --preset Cologne-v1

- Manually set a value in the base config:
  python compute_reward_factor.py --preset Cologne-v1 --value -250.0

Notes
- This script leaves the computation logic empty on purpose. Provide --value
  to set the factor, or implement `compute_reward_factor_for_preset()`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import re

import jax
from imp_act import make
import jax.numpy as jnp


SCRIPT_DIR = Path(__file__).resolve().parent


def compute_reward_factor_for_preset(preset_dir: Path) -> Optional[float]:
    """Estimate travel_time_reward_factor for a preset (fast heuristic).

    Method
    - Build the unconstrained JAX env ("<preset>-unconstrained-jax").
    - For each segment, apply a single corrective
      action with value 4, step once, and read from `info`:
        maintenance_reward and total_travel_time - base_total_travel_time.
    - Compute factor = -abs(mean(maintenance) / mean(delay) x 1.25), with
      simple scaling to keep numbers well-behaved.

    Inputs/assumptions
    - `preset_dir` points to the preset folder; its name matches the base YAML.
    - The unconstrained variant is registered and loadable via `make()`.

    Returns
    - Negative float on success; None if the computation cannot be performed.
    - This function does not write files; callers choose whether to persist.
    """
    try:
        preset_name = preset_dir.name
        # Use JAX env variant for speed and reproducibility: registered as "<name>-unconstrained-jax"
        env = make(f"{preset_name}-unconstrained-jax")

        SCALE = 1e8
        # Disable any existing reward factor to isolate effects
        env.travel_time_reward_factor = 0.0

        # RNG setup
        SEED = 998
        key = jax.random.PRNGKey(SEED)
        print(f"Computing reward factor for preset: {preset_name}")

        def compute_reward_elements(
            idx_segment: int, action_value: int
        ) -> tuple[float, float]:
            nonlocal key
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)

            actions = jnp.zeros((env.total_num_segments,), dtype=jnp.int32)
            actions = actions.at[idx_segment].set(jnp.int32(action_value))

            key, step_key = jax.random.split(key)
            obs, next_state, reward, done, info = env.step(step_key, state, actions)

            rew_maintenance = info["reward_elements"]["maintenance_reward"] / SCALE
            total_tt = info["total_travel_time"]
            base_tt = env.base_total_travel_time
            delay = (total_tt - base_tt) / SCALE
            return rew_maintenance, delay

        action_value = 4
        rew_maintenance_list = []
        delays_list = []
        for i in range(env.total_num_segments):
            rew_maintenance, delay = compute_reward_elements(i, action_value)
            rew_maintenance_list.append(rew_maintenance)
            delays_list.append(delay)

        rew_maint_mean = np.mean(rew_maintenance_list)
        rew_delay_mean = np.mean(delays_list)
        # Nicely formatted summary stats for quick inspection
        print(
            f"Mean maintenance reward: {rew_maint_mean:.4f}, "
            f"Mean delay: {rew_delay_mean:.4f}"
        )
        if abs(rew_delay_mean) < 1e-9:
            raise ValueError(
                f"Mean delays is {rew_delay_mean}; cannot compute reward factor safely."
            )
        raw_reward_factor = (rew_maint_mean / rew_delay_mean) * 1.25
        reward_factor = -abs(round(raw_reward_factor, 2))
        print(
            f"Reward factor: {rew_maint_mean:.4f}/{rew_delay_mean:.4f} ~ {reward_factor:.2f}"
        )
        return reward_factor
    except ValueError as e:
        # Known benign: zero (or near-zero) mean delay makes the ratio unstable.
        if "Mean delays is" in str(e):
            print("[reward-factor] Mean delay ~ 0; cannot compute a stable factor.")
            return None
        # Otherwise, let the precise ValueError propagate.
        raise
    except Exception as e:
        # Compact surface of unexpected failures with context.
        raise RuntimeError(
            f"[reward-factor] Unexpected error during computation: {type(e).__name__}: {e}"
        ) from e


def _resolve_preset_dir(preset: str) -> Path:
    """Resolve a preset name or path to a directory path.

    - If `preset` exists as a path, return it.
    - Otherwise, look under <repo>/imp-act/imp_act/environments/presets/<preset>.
    """
    p = Path(preset)
    if p.exists():
        return p
    return SCRIPT_DIR.parent / "presets" / preset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and set reward factor for a preset"
    )
    p.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Preset name (e.g., Cologne-v1) or full path to a preset directory",
    )
    p.add_argument(
        "--value",
        type=float,
        default=None,
        help="Manual reward factor to set. If omitted, attempts to compute.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute (or use --value) and print the factor without writing to the YAML",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve preset dir from name or full path
    preset_dir = _resolve_preset_dir(args.preset)
    if not preset_dir.exists() or not preset_dir.is_dir():
        raise ValueError(
            f"Preset directory does not exist or is not a directory: {preset_dir}. "
            f"Pass a valid name (e.g., Cologne-v1) or full path."
        )

    factor = args.value
    if factor is None:
        factor = compute_reward_factor_for_preset(preset_dir)
        if factor is None:
            raise ValueError(
                "Reward factor is not provided and the computation is not implemented. "
                "Pass --value or use a preset where computation succeeds."
            )

    # Ensure reward factor is strictly negative
    factor = float(factor)
    if factor == 0.0:
        raise ValueError("Reward factor must be non-zero and negative.")
    if factor > 0.0:
        factor = -abs(factor)

    # If dry-run, just exit
    if args.dry_run:
        return

    base_yaml = preset_dir / f"{preset_dir.name}.yaml"
    if not base_yaml.exists():
        raise ValueError(
            f"Base preset YAML not found: {base_yaml}. Ensure it exists (name must match folder)."
        )
    with open(base_yaml, "r") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("traffic", {})
    old = data["traffic"].get("travel_time_reward_factor")
    data["traffic"]["travel_time_reward_factor"] = float(factor)

    if old == factor:
        print("No changes were necessary (value already up-to-date).")
    else:
        # Dump and re-inline initial_damage_distribution as a one-line list if present
        yaml_text = yaml.safe_dump(data, sort_keys=False)
        try:
            idd = data.get("maintenance", {}).get("initial_damage_distribution")
            if isinstance(idd, (list, tuple)) and len(idd) == 5:
                idd_inline = ", ".join(str(x) for x in idd)
                pattern = (
                    r"^([ \t]*)initial_damage_distribution:\n(?:^[ \t]*-[^\n]*\n){5}"
                )
                yaml_text = re.sub(
                    pattern,
                    lambda m: f"{m.group(1)}initial_damage_distribution: [{idd_inline}]\n",
                    yaml_text,
                    flags=re.MULTILINE,
                )
            # Ensure a blank line between main sections for readability
            yaml_text = yaml_text.replace("\ntraffic:\n", "\n\ntraffic:\n")
            yaml_text = yaml_text.replace("\ntopology:\n", "\n\ntopology:\n")
        except Exception:
            pass
        with open(base_yaml, "w") as f:
            f.write(yaml_text)
        print(f"Updated: {base_yaml}")


if __name__ == "__main__":
    main()
