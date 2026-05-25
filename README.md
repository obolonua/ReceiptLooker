# ReceiptLooker

This is an assignment for the Algorithms and Artificial Intelligence course in the Computer Science bachelor’s program at the University of Helsinki.

## Usage

### Requirements

- Python 3.11
- Font files in `data/fonts/` (`.ttf` or `.ttc`)

### Install dependencies

This project uses Poetry.

```bash
poetry install
```

### Preview generated digits

Run the digit preview script to see sample synthetic images:

```bash
poetry run python generator/render_digits.py
```

### Generate the dataset

Run the dataset generator to create the `.npy` files in `data/generated/`:

```bash
poetry run python generator/create_dataset.py
```

The script writes:

- `data/generated/X_train.npy`
- `data/generated/X_val.npy`
- `data/generated/y_train.npy`
- `data/generated/y_val.npy`

### Run tests

```bash
poetry run python -m unittest discover -s tests
```

## Dokumentaatio
