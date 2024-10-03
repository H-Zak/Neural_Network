# Définir des variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Cible par défaut
.PHONY: all
all: install test

# Installation des dépendances
.PHONY: install
install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirement.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirement.txt


# Exécuter les tests
.PHONY: test
test:
	$(PYTHON) -m unittest discover test

# Nettoyer les fichiers temporaires
.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

# Supprimer l'environnement virtuel
.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: update-requirements
update-requirements: $(VENV)/bin/activate
	$(PIP) freeze > requirement.txt

.PHONY: run
run: $(VENV)/bin/activate
	$(PYTHON) -m neural_network.main