#!/bin/sh
set -e

run_with_probability() {
    if [ $((RANDOM % 1)) -eq 0 ]; then
        python "$1"
    fi
}

for dir in docstrings_examples/*/
do
    if [ "$(basename "$dir")" != "__pycache__" ]; then
        for file in "$dir"*.py
        do
            if [ -f "$file" ]; then
                run_with_probability "$file"
            fi
        done
    fi
done

for file in readme_examples/*.py
do
    run_with_probability "$file"
done

for file in getting_started/*.py
do
    run_with_probability "$file"
done
