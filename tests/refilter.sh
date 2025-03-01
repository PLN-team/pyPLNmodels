#!/bin/sh

rm docstrings_examples/models/*.py

python _create_readme_getting_started_and_docstrings_tests.py

cd docstrings_examples/models/

./run.sh
