#!/bin/sh
set -e

for dir in docstrings_examples/*/
do
    if [ "$(basename "$dir")" != "__pycache__" ]; then
        for file in "$dir"*.py
        do
            if [ -f "$file" ]; then
                python "$file"
            fi
        done
    fi
done

for file in readme_examples/*.py
do
    python "$file"
done

for file in getting_started/*.py
do
    python "$file"
done
