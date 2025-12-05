# Datasets

## 1. Road Network and Traffic Data

The required dataset can be downloaded [here](https://data.mendeley.com/datasets/py2zkrb65h/1). Please put the files in the `data` folder.

The related publication "Synthetic European road freight transport flow data" by Daniel Speth, Verena Sauter, Patrick Plötz, Tim Signer" can be found [here](https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content)

It is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## 2. Deterioration and Cost Models

Our deterioration and maintenance cost models are adapted from:

M. Saifullah, K.G. Papakonstantinou, C.P. Andriotis, S.M. Stoffels, Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management, 2024. [https://arxiv.org/abs/2401.12455](https://arxiv.org/abs/2401.12455)

Concretely:
- We adopt a discrete condition‑state model for pavement components with Markovian deterioration, using transition matrices that depend on:
  - current condition state, and
  - applied maintenance action (e.g., Do Nothing, Minor Repair, Major Repair, Reconstruction).
- Maintenance actions and their qualitative effects mirror those in the paper:
  - Do Nothing — no direct cost, continued deterioration.
  - Minor Repair — low cost, modest improvement / slowdown in deterioration.
  - Major Repair — higher cost, substantial condition improvement and reduced effective age.
  - Replacement — highest cost, restores the asset to an “as‑new” state with reset deterioration rate.
- Unit costs per action are based on the per‑area cost ranges reported in the paper (USD/m² for different road classes) and rescaled/aggregated for the segments in our network.

In summary, the network + traffic comes from Speth et al., while the stochastic deterioration dynamics and action cost structure follow the transportation infrastructure management framework of Saifullah et al., adapted to our European road freight testbed.

# Environment Creation Workflow

This section describes the steps to create a large-scale IMP-ACT environment from the road network and traffic data. The workflow consists of four main steps:

1. Preprocess traffic data — fix trip edge‑path directions so O/D match.
   Produces `data/01_Trucktrafficflow_fixed.csv` (created automatically on first run if missing).
2. Create large graph — build the network and export a preset using Hydra.
   Writes graph files and a base/unconstrained/only‑maintenance preset under `presets/<name>/`.
3. Compute reward factor — set the traffic `travel_time_reward_factor` after export.
   Provide a value or compute it; only the base preset is updated so variants inherit it.
4. Create budget scenarios — estimate budgets from rollouts and write variants.
   `limited-budget` updates the base preset in place; other labels are written as separate files.

All steps are config‑driven and reproducible; the provided defaults run out‑of‑the‑box.
Reward factor and budget computation are decoupled from graph export so presets stay reusable and easy to version.

## 1. Preprocessing the Traffic Data
We validate and correct trip edge-path directions in `01_Trucktrafficflow.csv` so they align with each trip's origin and destination. The script writes the corrected file `01_Trucktrafficflow_fixed.csv` back to the `data/` folder.

- Automatic: running `create_large_graph.py` (without `--skip-traffic`) will create `data/01_Trucktrafficflow_fixed.csv` on first run if it is missing.
- Manual (optional):
  ```bash
  python imp_act/environments/dev/fix_traffic_paths.py --fix
  ```
  By default, the fixed file is saved to `<script_dir>/data/01_Trucktrafficflow_fixed.csv`.

## 2. Create Large Graph (config‑driven)
This script uses a Hydra YAML config: `imp-act/imp_act/environments/dev/create_large_graph_config.yaml`.

- Edit the config to set either `country: DE|ALL` or `coordinate_range: [min_x,max_x,min_y,max_y]`.
- Run with Hydra overrides if needed:
  - `python imp_act/environments/dev/create_large_graph.py country=DE`
  - `python imp_act/environments/dev/create_large_graph.py coordinate_range=[6.5,7.5,50.5,51.5]`
- Optional flags in config: `skip_traffic`, `validate`, `export_preset`, `preset_name`.

Preset export
- If `export_preset: true`, the script copies `graph.graphml`, `segments.csv`, `traffic.csv` and writes preset configs into `imp-act/imp_act/environments/presets/<preset_name>/`:
  - `<preset_name>.yaml` (base), `<preset_name>-unconstrained.yaml`, `<preset_name>-only-maintenance.yaml`.
- The `travel_time_reward_factor` is not set at creation time; compute it later (see below).

### Outputs
Results are written under `--output-dir` (default: `<script_dir>/output`) into a scope-specific subfolder:
- Country scope: `countries/<COUNTRY_CODE>` (e.g., `countries/DE`)
- Coordinate range scope: `coordinate_ranges/<min_x>_<max_x>_<min_y>_<max_y>`

Each scope contains:
- `graph_full.graphml`, `graph.graphml`, `new-edges.yaml`
- `segments.csv`, `traffic_full.csv`, `traffic.csv`
- `info.yaml`, `network.yaml`

## 3. Compute Reward Factor (separate step)
Use the helper to set `traffic.travel_time_reward_factor` in a preset after creation. It only updates the base preset YAML; variants inherit it.

- Set value in base config:
  - `python imp_act/environments/dev/compute_reward_factor.py --preset <preset_name> --value -250.0`

- Compute (stub):
  - `python imp_act/environments/dev/compute_reward_factor.py --preset <preset_name>`

Note: The computation function is intentionally left empty; pass `--value` or implement it in `compute_reward_factor_for_preset`.

## 4. Create Budget Scenarios
Compute per‑period maintenance‑cost quantiles from heuristic rollouts and map them to named budgets. Uses `create_budget_scenarios_config.yaml` by default; `limited-budget` updates the base preset while other labels are written as separate files.

- Dry run (print budgets only, uses config defaults):
  ```bash
  python imp_act/environments/dev/create_budget_scenarios.py
  ```

- Override preset; write variants and overwrite if they exist:
  ```bash
  python imp_act/environments/dev/create_budget_scenarios.py \
    preset=Cologne-v1 dry_run=false overwrite=true
  ```
