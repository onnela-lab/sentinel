.PHONY : clean data

DATA_PATH ?= data/project_tycho_processed_cases.csv
EXECUTE_NB = cd $(dir $<) \
	&& jupytext --to ipynb --output - $(notdir $<) \
	| jupyter nbconvert --stdin --execute --to html --output $(notdir $@)

data : data/ProjectTycho_Level2_v1.1.0.csv data/project_tycho_processed_cases.csv

data/ProjectTycho_Level2_v1.1.0.csv : data/ProjectTycho_Level2_v1.1.0.zip
	unzip -o -d $(dir $@) $<
# Touch the file to update the timestamp and show the first few lines.
	touch $@
	head $@

data/ProjectTycho_Level2_v1.1.0.zip :
	mkdir -p $(dir $@)
	curl -L -o $@ https://zenodo.org/records/12608994/files/ProjectTycho_Level2_v1.1.0.zip?download=1

data/project_tycho_processed_cases.csv : notebooks/preprocessing.md data/ProjectTycho_Level2_v1.1.0.csv
	${EXECUTE_NB}

results : ${DATA_PATH} configs/default.py
	python -m sentinel.fit --config configs/default.py --data ${DATA_PATH} workspace/default/

clean :
	rm -rf data workspace
