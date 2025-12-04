"""
Compute and set the travel_time_reward_factor for a preset.

Usage examples
- Compute (stubbed) and update the base config:
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

from imp_act import make

SCRIPT_DIR = Path(__file__).resolve().parent


def compute_reward_factor_for_preset(preset_dir: Path) -> Optional[float]:
    """Placeholder for computing travel_time_reward_factor for a preset.

    Implement your logic here (e.g., run a heuristic policy for rollouts and
    compute the ratio as described). Return a float or None if unavailable.
    """
    preset_name = preset_dir.name
    env = make(preset_name)

    # Match notebook scaling
    env.budget_amount = env.budget_amount * 1e8
    env.travel_time_reward_factor = 0.0

    def compute_reward_elements(idx_edge: int, action_value: int) -> tuple[float, float]:
        """Return (maintenance_reward, delay) for a single-edge action."""
        _ = env.reset()
        actions = [[0 for _ in edge["road_edge"].segments] for edge in env.graph.es]
        actions[idx_edge][0] = action_value

        _ = env.reset()
        _, _, _, info = env.step(actions)

        rew_maintenance = info["reward_elements"]["maintenance_reward"] / 1e8
        delays = (info["total_travel_time"] - env.base_total_travel_time) / 1e8
        return rew_maintenance, delays

    # Corrective replacements (action_value=4)
    rew_maintenance_list = []
    delays_list = []
    for i in range(len(env.graph.es)):
        rew_maintenance, delays = compute_reward_elements(i, 4)
        rew_maintenance_list.append(rew_maintenance)
        delays_list.append(delays)

    rew_maintenance_mean = float(np.mean(rew_maintenance_list))
    rew_delays_mean = float(np.mean(delays_list))

    if abs(rew_delays_mean) < 1e-9:
        raise ValueError("Mean delays is ~0; cannot compute reward factor safely.")

    reward_factor = round((rew_maintenance_mean / rew_delays_mean) * 1.25, 2)
    return -abs(reward_factor)
    return None


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
                "Pass --value to set it explicitly."
            )

    # Ensure reward factor is strictly negative
    factor = float(factor)
    if factor == 0.0:
        raise ValueError("Reward factor must be non-zero and negative.")
    if factor > 0.0:
        factor = -abs(factor)

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
                pattern = r"^([ \t]*)initial_damage_distribution:\n(?:^[ \t]*-[^\n]*\n){5}"
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
