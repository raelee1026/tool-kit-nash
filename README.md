# NASH Eval

NASH Eval is a self-contained Python toolkit for NASH scoring, NumFinE
evaluation, and local visualization of exported web-demo JSON results.

## Install

```bash
pip install nash-eval
```

For local development:

```bash
pip install -e .
```

## Launch The Local Viewer

Open the website only:

```bash
nash-eval serve
```

Open the website with a completed result JSON:

```bash
nash-eval serve results.json
```

The command starts a local HTTP server, serves the bundled React frontend, and
opens the browser. When `results.json` is provided, it is loaded in Batch
Upload.

You can also launch it from Python:

```python
import nash

nash.launch()
nash.launch("results.json")
```

## Python API

```python
import nash

dataset = nash.load_dataset("numfine_triplet", split="test")
result = nash.evaluate(
    dataset,
    protocol="triplet",
    model="sentence-transformers/all-MiniLM-L6-v2",
    limit=1,
)
nash.export_web_json(result, "results.json")
```

Supported datasets:

```text
numfine_triplet
numfine_crosspair
numfine_listwise
```

Bundled data files:

```text
nash/data/triplet_test.json
nash/data/crosspair_test.json
nash/data/listwise_test.json
```

Supported protocols:

```text
triplet
crosspair
listwise
```
