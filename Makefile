.PHONY: install test figures paper-figures synthetic-figures clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

paper-figures:
	python scripts/plot_paper_figures.py

synthetic-figures:
	python scripts/plot_pipeline_example.py
	python scripts/run_consensus_experiment.py
	python scripts/run_fusion_experiment.py
	python scripts/run_spatial_map.py
	python scripts/run_coop_cyclo_experiment.py
	python scripts/run_temporal_campaign.py

figures: paper-figures synthetic-figures

clean:
	rm -rf results/npz .pytest_cache **/__pycache__
