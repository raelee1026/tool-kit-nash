# NASH Eval

NASH Eval is a Python toolkit for NASH scoring, NumFinE evaluation, and local
visualization of completed evaluation result JSON files.

The package includes:

- the NASH Python evaluator
- bundled NumFinE JSON data
- a built React visualization frontend served locally by the CLI

## Install

Install from GitHub:

```bash
pip install git+https://github.com/raelee1026/tool-kit-nash.git
```

For local development from this repository:

```bash
pip install -e .
```

## Quick Start

Create a web-demo-compatible result JSON from the bundled NumFinE triplet data:

```python
import nash

dataset = nash.load_dataset("numfine_triplet", split="easy")
result = nash.evaluate(
    dataset,
    protocol="triplet",
    model="sentence-transformers/all-MiniLM-L6-v2",
)
nash.export_web_json(result, "results.json")
```

Open the local viewer with that result:

```bash
nash-eval serve results.json
```

Open the website only:

```bash
nash-eval serve
```

Python launcher:

```python
import nash

nash.launch()
nash.launch("results.json")
```

## Built-In Datasets

Supported dataset names:

```text
numfine_triplet
numfine_crosspair
numfine_listwise
```

Bundled files:

```text
nash/data/triplet_easy.json
nash/data/triplet_medium.json
nash/data/triplet_hard.json
nash/data/triplet_test.json
nash/data/crosspair.json
nash/data/crosspair_test.json
nash/data/listwise.json
nash/data/listwise_test.json
```

Supported splits:

```python
nash.load_dataset("numfine_triplet", split="easy")
nash.load_dataset("numfine_triplet", split="medium")
nash.load_dataset("numfine_triplet", split="hard")
nash.load_dataset("numfine_triplet", split="test")
nash.load_dataset("numfine_triplet", split="all")

nash.load_dataset("numfine_crosspair", split="all")
nash.load_dataset("numfine_crosspair", split="test")

nash.load_dataset("numfine_listwise", split="all")
nash.load_dataset("numfine_listwise", split="test")
```

## Custom Local Dataset

You can load your own local JSON file if it follows the same task format:

```python
dataset = nash.load_dataset("/path/to/my_triplet.json")
result = nash.evaluate(dataset, protocol="triplet")
```

Triplet rows should contain:

```json
{
  "sentence_a": {"text": "anchor sentence"},
  "sentence_b": {"text": "s1 sentence"},
  "sentence_c": {"text": "s2 sentence"}
}
```

Cross-pair rows should contain:

```json
{
  "sentence_a": {"text": "pair A sentence 1"},
  "sentence_b": {"text": "pair A sentence 2"},
  "sentence_c": {"text": "pair B sentence 1"},
  "sentence_d": {"text": "pair B sentence 2"}
}
```

Listwise rows should contain:

```json
{
  "sentence_a": {"text": "anchor sentence"},
  "candidates": [
    {"text": "candidate 1"},
    {"text": "candidate 2"}
  ]
}
```

## Pair Scoring

```python
import nash

row = nash.score(
    "EPS rose 10% to $11.",
    "EPS was $10 in 2024, a 25% increase.",
    model="sentence-transformers/all-MiniLM-L6-v2",
)

print(row["baseline_score"])
print(row["nash_score"])
print(row["numeric_score"])
print(row["alignment"]["matrix"])
```

For BERTScore-style models:

```python
row = nash.score(
    "EPS rose 10% to $11.",
    "EPS was $10 in 2024, a 25% increase.",
    model="bert-base-uncased",
    backend="bertscore",
)
```

## Evaluation Protocols

```python
triplet = nash.evaluate(
    nash.load_dataset("numfine_triplet", split="easy"),
    protocol="triplet",
)

crosspair = nash.evaluate(
    nash.load_dataset("numfine_crosspair", split="all"),
    protocol="crosspair",
)

listwise = nash.evaluate(
    nash.load_dataset("numfine_listwise", split="all"),
    protocol="listwise",
)
```

Export any evaluation result for the local web viewer:

```python
nash.export_web_json(triplet, "triplet_results.json")
```

## Build Demo Result JSON

Triplet:

```python
import nash

dataset = nash.load_dataset("numfine_triplet", split="easy")
result = nash.evaluate(dataset, protocol="triplet")
nash.export_web_json(result, "triplet_results.json")
```

Cross-pair:

```python
import nash

dataset = nash.load_dataset("numfine_crosspair", split="all")
result = nash.evaluate(dataset, protocol="crosspair")
nash.export_web_json(result, "crosspair_results.json")
```

Listwise:

```python
import nash

dataset = nash.load_dataset("numfine_listwise", split="all")
result = nash.evaluate(dataset, protocol="listwise")
nash.export_web_json(result, "listwise_results.json")
```

Open any generated JSON:

```bash
nash-eval serve triplet_results.json
```

## Build The Bundled Web Demo

The published package already includes built static files in `nash/web/`.

If you modify the React source in `web/`, rebuild and copy the static files:

```bash
cd web
npm install
npm run build
cd ..
rm -rf nash/web
mkdir -p nash/web
cp -R web/dist/. nash/web/
```

Then reinstall locally:

```bash
pip install -e .
```

## Package Metadata

`pyproject.toml` contains the package version. When releasing a new package
build, bump the version normally.
