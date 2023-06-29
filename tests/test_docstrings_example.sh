search_dir="docstrings_examples"
for entry in "$search_dir"/*
do
  python "$entry"
done
