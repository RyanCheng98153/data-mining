# News Recommendation Prediction

## Preparation
- Put your competition data in the `train` and `test` directory ``

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

### Download the required resources
```bash
sh download.sh
```

## Usage

Run the main script:
```bash
torchrun --nproc_per_node=2 hw3_314554025.py
```

> You can `change nproc_per_node=` according to your gpu resources
