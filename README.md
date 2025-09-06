# Multi-Agent Data Collection (Simulated)

## Quick Start
#bash
python -m pip install -r requirements.txt
make run-demo
make load
# Full grid (heavy):
make run-full
```

#Or directly in Python
python3 -m pip install -r requirements.txt
python3 src/main.py --manifest configs/factors.json --replications 3 --out data/logs --seed 1234
python3 src/load_parquet.py --path data/logs

