PY=python
OUT=data/logs

.PHONY: setup run-demo run-full load clean

setup:
	$(PY) -m pip install -r requirements.txt

run-demo:
	$(PY) src/main.py --manifest configs/factors.json --replications 3 --out $(OUT) --seed 1234

run-full:
	$(PY) src/main.py --manifest configs/factors_full.json --replications 20 --out $(OUT) --seed 4242 --chunksize 20000

load:
	$(PY) src/load_parquet.py --path $(OUT)

clean:
	rm -rf data/logs/*.parquet configs/run_manifest.jsonl
