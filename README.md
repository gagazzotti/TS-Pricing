# Option pricing with Mellin series under Tempered Stable process 

[![Python badge](https://img.shields.io/badge/Python-3.11.11-0066cc?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-31111/)
[![Pylint badge](https://img.shields.io/badge/Linting-pylint-brightgreen?style=for-the-badge)](https://pylint.pycqa.org/en/latest/)
[![Ruff format badge](https://img.shields.io/badge/Formatter-Ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/formatter/)

The code allows to calibrate to price European options with Mellin expansion technique.

...

## Installation
A script is available for easy installation of dependencies 
```bash
source install.bash
```


## How to use

...

## Results

...

## Tests

...

## Interesting ? 

...

## Tested on

[![Ubuntu badge](https://img.shields.io/badge/Ubuntu-24.04-cc3300?style=for-the-badge&logo=ubuntu)](https://www.releases.ubuntu.com/24.04/)
[![Conda badge](https://img.shields.io/badge/conda-24.9.2-339933?style=for-the-badge&logo=anaconda)](https://docs.conda.io/projects/conda/en/24.9.x/)
[![Intel badge](https://img.shields.io/badge/CPU-%20i5_10210U%201.60GHZ-blue?style=for-the-badge&logo=intel)](https://www.intel.com/content/www/us/en/products/sku/195436/intel-core-i510210u-processor-6m-cache-up-to-4-20-ghz/specifications.html)


## Todo

- check that the cpp code gamma_incomp pour upper gamma et upper gamma_vect is the right one (recompile and check)
- check that the conda env is enough to generate the code (add +git@https://...)
- do a toy example
- do the short time behaviour
- remove tests/test_vect.py, ./test.py
- faire computational time comme mtn mais avec différent $N$ pour les deux
- normal que ce soit à 10-13 et pas 10-15 (on regarde en erreur relative!!!!!!!!)
- faire un test pour les densités (éloignés de 0....)


- nom des fichiers snake case
- vérifier que les formules sont valides pour différents temps
- enlever les TBD (Ctrl+F)
- regarder les todo
- implement itm, atm
- BG prices with T
- 3.corriger le nom alpha_m dans proj pour BG
