# Using a multivalent binding model to infer antibody Fc species from systems serology data

This model helps us learn relationships between antibody Fc structural features
and immune receptor interaction, effector cell recruitment, and disease outcome.
See our
[manuscript](https://www.biorxiv.org/content/10.1101/2024.07.05.602296v1).

[![Test](https://github.com/meyer-lab/mechanismSerology/actions/workflows/pytest.yml/badge.svg)](https://github.com/meyer-lab/mechanismSerology/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/meyer-lab/mechanismSerology/branch/master/graph/badge.svg)](https://codecov.io/gh/meyer-lab/mechanismSerology)

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency
management.

You can install our project as a dependency by adding the following to your `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
maserol = { git = "https://github.com/meyer-lab/mechanismSerology.git", branch = "main" }
```

Otherwise, you can clone the repository and install the dependencies by running the following commands:

```bash
git clone https://github.com/meyer-lab/mechanismSerology.git
cd mechanismSerology
poetry install
```

## Running the code

### Figure generation
The figures can be generated using:

```bash
poetry run make all
```

or for a specific figure:

```bash
poetry run make output/figure_X.svg
```

### Using the model

The model can be used without any fine-tuning on new systems serology datasets.
The model uses numerical optimization to infer its outputs and this is handled
by the `optimize_loss` function.

```python
from maserol.core import optimize_loss

# load data ...

# run inference
opts = assemble_options(data)
x, ctx = optimize_loss(data, **opts, return_reshaped_params=True)
# x contains the inferred parameters, including the inferred antibody abundances (as "Rtot")

# if you want the inferences as a pandas DataFrame
Rtot = Rtot_to_df(x["Rtot"], data, rcps=list(opts["rcps"]))
```

### Using our datasets

All of our datasets can be accessed through the `maserol.datasets` module.

```python
from maserol.datasets import Zohar, Kaplonek

zohar = Zohar()

zohar_data = zohar.get_detection_signal()

zohar_meta = zohar.get_metadata()
```

