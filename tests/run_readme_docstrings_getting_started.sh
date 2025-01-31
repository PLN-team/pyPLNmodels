#!/bin/sh
for file in docstrings_examples/*
do
    python $file
done

for file in docstrings_examples/test_load_data/*
do
    python $file
done

for file in readme_examples/*
do
    python $file
done

for file in getting_started/*
do
    python $file
done
