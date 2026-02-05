.PHONY: all clean data fit results

DATA_PATH ?= data/project_tycho_processed_cases.csv
CONFIG ?= configs/helmert.py
OUTPUT ?= workspace/helmert

EXECUTE_NB = cd $(dir $<) \
	&& jupytext --to ipynb --output - $(notdir $<) \
	| jupyter nbconvert --stdin --execute --to html --output $(notdir $@)

all: data fit results

data: ${DATA_PATH}

data/ProjectTycho_Level2_v1.1.0.csv: data/ProjectTycho_Level2_v1.1.0.zip
	unzip -o -d $(dir $@) $<
	touch $@
	head $@

data/ProjectTycho_Level2_v1.1.0.zip:
	mkdir -p $(dir $@)
	curl -L -o $@ https://zenodo.org/records/12608994/files/ProjectTycho_Level2_v1.1.0.zip?download=1

${DATA_PATH}: notebooks/preprocessing.md data/ProjectTycho_Level2_v1.1.0.csv
	${EXECUTE_NB}

fit: ${OUTPUT}/final.pkl

${OUTPUT}/final.pkl: ${DATA_PATH} ${CONFIG}
	python -m sentinel.fit --config ${CONFIG} --data ${DATA_PATH} ${OUTPUT}

results: notebooks/results.html

notebooks/results.html: notebooks/results.md ${OUTPUT}/final.pkl
	${EXECUTE_NB}

clean:
	rm -rf data workspace notebooks/*.html
