# Contributing to ChurnSense

## Development Setup

```bash
git clone https://github.com/pranjalts07/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

## Before Pushing Large Changes

This repo contains binary files (trained model, parquet data, PNG plots). If you get HTTP 400 errors when pushing, increase git's HTTP buffer:

```bash
git config http.postBuffer 524288000
```

This sets the buffer to 500MB (from the default 1MB), allowing large binary files to push successfully.

## Running Tests

```bash
pytest tests/ -v
```

## Training a New Model

```bash
python src/train.py
```

## License

MIT License - see LICENSE file
