#!/bin/bash
# cd ../../artifact_evaluation/fig7/
# python run_orion.py > 123.log
# python exe_prof.py
# python avg_prof.py
# python merge.py

cd ~/orion
bash compile.sh
python run_orion.py > 123.log
python exe_prof.py
python avg_prof.py