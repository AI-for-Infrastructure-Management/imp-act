# Dataset
The required dataset can be downloaded [here](https://data.mendeley.com/datasets/py2zkrb65h/1). Please put the files in the `data` folder.

The related publication can be found [here](https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content)

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
