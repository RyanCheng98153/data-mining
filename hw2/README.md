# Linear Regression Model 

## Preparation
- Put your `train.csv` and `test.csv` to the local directory with `train_model.py`

## Installation (uv or pip)

### Using [uv](https://github.com/astral-sh/uv) (recommended)
```bash
uv sync
```
Activate the virtual environment:
- On Windows:
    ```bash
    ./.venv/Scripts/Activate.ps1
    ```
- On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```

### Using pip (if does not have uv)
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python train_model.py --train ./train.csv
```
