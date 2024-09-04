# phonotactic-sensitivity

Code and materials for our Interspeech 2024 paper on 'Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0'

## To use this repository:
1. Clone it locally:
   `git clone git@github.com:mdhk/phonotactic-sensitivity.git`
2. Configure the virtual environment — with [Poetry installed](https://python-poetry.org/docs/#installing-with-the-official-installer), run the following commands in this repository's root directory:
    ```
    poetry config virtualenvs.in-project true
    poetry env use python3.10
    poetry install
    ```
    This will install dependencies and configure the environment as specified in [`pyproject.toml`](https://github.com/mdhk/phonotactic-sensitivity/blob/main/pyproject.toml).
3. (Re)compute measures by running `poetry run python scripts/compute_measures.py`.