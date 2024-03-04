# Dataset
The required dataset can be downloaded from:
"https://data.mendeley.com/datasets/py2zkrb65h/1"

The related paper can be found under:
"https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content"

# Running the export
To run the export, execute the following command after downloading the dataset and placing it in the `data` folder:

```bash
python environments/graph/create_large_graph.py -c {COUNTRY_CODE}
```

To export a single country, or use:

```bash
python environments/graph/create_large_graph.py -c ALL
```

To export all countries.

To skip the traffic filtering which can take a long time, use the `--skip-traffic` flag:

```bash
python environments/graph/create_large_graph.py -c ALL --skip-traffic
```

For more information on the available flags, use the `--help` flag:

```bash
python environments/graph/create_large_graph.py --help
```
