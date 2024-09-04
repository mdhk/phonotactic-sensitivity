# phonotactic-sensitivity

Code and materials for our Interspeech 2024 paper on ['Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0'](https://doi.org/10.21437/Interspeech.2024-2490).

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

## Citation
The paper can be cited as follows:

de Heer Kloots, M. & Zuidema, W. (2024). Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0. _Proc. Interspeech 2024_, 4593-4597. doi: [10.21437/Interspeech.2024-2490](https://doi.org/10.21437/Interspeech.2024-2490).
```
@inproceedings{deheerkloots24_interspeech,
  title     = {Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0},
  author    = {Marianne {de Heer Kloots} and Willem Zuidema},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4593--4597},
  doi       = {10.21437/Interspeech.2024-2490},
}
```