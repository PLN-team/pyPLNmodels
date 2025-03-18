#!/bin/sh

rm docstrings_examples/models/*.py

python _create_readme_getting_started_and_docstrings_tests.py

cd docstrings_examples/models/

output_file="output.txt"

# Clear the output file if it exists
> "$output_file"

# Traverse directories and subdirectories
find . -name "*.py" | while read -r file; do
    # Run the Python file
    if ! python3 "$file" 2>> "$output_file"; then
        # If there is an error, log it to output.txt
        echo "Error in $file" >> "$output_file"
    else
        # If no error, remove the file
        rm "$file"
    fi
done
