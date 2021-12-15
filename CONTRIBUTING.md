# Install in dev mode

1. Create a virtual environment

```
python -m venv .venv
```

2. Activate the venv.

On Linux:
```
python3 -m venv .venv
. .venv/bin/activate
```

On Windows:
```
.venv\Scripts\activate
```

3. Install dev dependencies and then the package in editable mode:
```
pip install -r requirements/dev.txt && pip install -e .
```

# Run tests

To run the test suite, do

```
pytest
```

To run the whole tox testing party, do

```
tox
```


# Specifying dev and testing requirements using  `pip-compile`

`pip-compile` is used to transform `requirements/*.in` to requirement lists `requirements/*.txt`. `pip-compile` is part of pip-tools: <https://github.com/jazzband/pip-tools>

To add a dev or testing requirement, add the requirement to the `.in` file and then do
```
pip-compile requirements/{dev|test|whatever}.in
```

# Format code using `black`

Because `black` is the new black ;)

<https://black.readthedocs.io/en/stable/>


# Build

```
python -m build
```

# Release on PyPI

```
python -m twine upload --repository pypi dist/*
```

See also <https://packaging.python.org/tutorials/packaging-projects/>.
