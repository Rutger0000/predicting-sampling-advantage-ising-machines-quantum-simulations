#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ultrafast-full-boltzmann-machine
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
JULIA_INTERPRETER = julia --project=.  
ID = 0

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

.PHONY: activate
activate:
	source .venv/bin/activate

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



## Convert the independent weights to full weights
.PHONY: convert_weights
convert_weights:
	$(JULIA_INTERPRETER) ultrafast_full_boltzmann_machine/julia/weightconverter.jl

## Promote weights to exchange matrix and bias vector of Ising model
.PHONY: isingweights
isingweights:
	$(PYTHON_INTERPRETER) ultrafast_full_boltzmann_machine/boltzmannweights.py


## Generate configuration files
.PHONY: generate_configs
generate_configs: models/all_weights_converted.tsv 

models/all_weights_converted.tsv: models/computed_configurations/converted/converted_configurations.py ultrafast_full_boltzmann_machine/models/models.py
	$(PYTHON_INTERPRETER) models/computed_configurations/converted/converted_configurations.py

## Run MH sampling, Chromatic Gibbs sampling, and calculate autocorrelation time
.PHONY: example
example:
	$(JULIA_INTERPRETER) ultrafast_full_boltzmann_machine/julia/groundstate/sampling.jl
	$(JULIA_INTERPRETER) ultrafast_full_boltzmann_machine/julia/gibbs/example.jl
	$(PYTHON_INTERPRETER) ultrafast_full_boltzmann_machine/autocorrelation/autocorrelation.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
