# Définir des variables
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Cible par défaut
.PHONY: all
all: install lab

.PHONY: lab
lab:
	@$(PYTHON) -m lab.main

.PHONY: run
run:
	@$(PYTHON) -m neural_network.main

# Installation des dépendances
.PHONY: install
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Exécuter les tests
.PHONY: test
test:
	$(PYTHON) -m unittest discover tests

# Nettoyer les fichiers temporaires
.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

# Supprimer l'environnement virtuel
.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)
