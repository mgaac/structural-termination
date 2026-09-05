CONDA_ENV ?= structural-termination
RUN = conda run --no-capture-output -n $(CONDA_ENV)

.PHONY: env test reproduce smoke data train locked-prepare locked locked-no-supervision locked-compare locked-plot locked-summary

env:
	conda env create -f environment.yml

test:
	$(RUN) python -m pytest -q

reproduce:
	$(RUN) python -m src.reproduce latest

smoke:
	$(RUN) python -m src.reproduce smoke

data:
	$(RUN) python -m src.data.dataset --preset --seed 42

train:
	$(RUN) python -m src.train --config configs/termination.yaml

locked-prepare:
	$(RUN) python -m src.analysis.locked_termination_experiment --prepare-only

locked:
	$(RUN) python -m src.analysis.locked_termination_experiment

locked-no-supervision:
	$(RUN) python -m src.analysis.locked_termination_experiment \
		--config configs/locked_termination_no_supervision.yaml \
		--output-dir results/locked_multiseed_no_supervision

locked-compare:
	$(RUN) python -m src.analysis.compare_termination_supervision

locked-plot:
	$(RUN) python -m src.analysis.plot_latent_trajectory_pca

locked-summary:
	$(RUN) python -m src.analysis.plot_locked_results
