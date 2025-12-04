# Dataset
The required dataset can be downloaded [here](https://data.mendeley.com/datasets/py2zkrb65h/1). Please put the files in the `data` folder.

The related publication "Synthetic European road freight transport flow data" by Daniel Speth, Verena Sauter, Patrick Plötz, Tim Signer" can be found [here](https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content)

It is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

# Preprocessing the traffic data
We validate and correct trip edge-path directions in `01_Trucktrafficflow.csv` so they align with each trip's origin and destination. The script writes the corrected file `01_Trucktrafficflow_fixed.csv` back to the `data/` folder.

- Automatic: running `create_large_graph.py` (without `--skip-traffic`) will create `data/01_Trucktrafficflow_fixed.csv` on first run if it is missing.
- Manual (optional):
  ```bash
  python imp_act/environments/dev/fix_traffic_paths.py --fix
  ```
  By default, the fixed file is saved to `<script_dir>/data/01_Trucktrafficflow_fixed.csv`.

## Create Large Graph (config-driven)
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

## Outputs
Results are written under `--output-dir` (default: `<script_dir>/output`) into a scope-specific subfolder:
- Country scope: `countries/<COUNTRY_CODE>` (e.g., `countries/DE`)
- Coordinate range scope: `coordinate_ranges/<min_x>_<max_x>_<min_y>_<max_y>`

Each scope contains:
- `graph_full.graphml`, `graph.graphml`, `new-edges.yaml`
- `segments.csv`, `traffic_full.csv`, `traffic.csv`
- `info.yaml`, `network.yaml`
