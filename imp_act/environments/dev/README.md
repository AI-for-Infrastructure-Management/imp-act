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

# Running the export
To run the export, execute the following command after downloading the dataset and placing it in the `data` folder:

```bash
python imp_act/environments/dev/create_large_graph.py --coordinate-range {X_min} {X_max} {Y_min} {Y_max}
```

For example:

```bash
python imp_act/environments/dev/create_large_graph.py --coordinate-range 6.5 7.5 50.5 51.5
```

To export a single country use:
```bash
python imp_act/environments/dev/create_large_graph.py -c {COUNTRY_CODE}
```

For example:
```bash
python imp_act/environments/dev/create_large_graph.py -c BE
```

To export all countries, use:

```bash
python imp_act/environments/dev/create_large_graph.py -c ALL
```

To skip the traffic filtering which can take a long time, use the `--skip-traffic` flag:

```bash
python imp_act/environments/dev/create_large_graph.py -c ALL --skip-traffic
```

For more information on the available flags, use the `--help` flag:

```bash
python imp_act/environments/dev/create_large_graph.py --help
```

## Outputs
Results are written under `--output-dir` (default: `<script_dir>/output`) into a scope-specific subfolder:
- Country scope: `countries/<COUNTRY_CODE>` (e.g., `countries/DE`)
- Coordinate range scope: `coordinate_ranges/<min_x>_<max_x>_<min_y>_<max_y>`

Each scope contains:
- `graph_full.graphml`, `graph.graphml`, `new-edges.yaml`
- `segments.csv`, `traffic_full.csv`, `traffic.csv`
- `info.yaml`, `network.yaml`
