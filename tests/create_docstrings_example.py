import ast
import os


def get_lines(filename, filetype=".py"):
    with open(f"../pyPLNmodels/{filename}{filetype}") as file:
        lines = [line.rstrip() for line in file]
    return lines


def get_examples_docstring(lines):
    examples = []
    in_example = False
    example = []
    for line in lines:
        line = line.lstrip()
        if len(line) > 3:
            if line[0:3] == ">>>":
                in_example = True
                example.append(line[4:])
            else:
                if in_example is True:
                    examples.append(example)
                    example = []
                in_example = False
    return examples


def write_examples(examples, filename, dirname):
    for i in range(len(examples)):
        example = examples[i]
        nb_example = str(i + 1)
        example_filename = f"{dirname}/{prefix_filename}_example_{nb_example}.py"
        try:
            os.remove(example_filename)
        except FileNotFoundError:
            pass
        with open(example_filename, "a") as the_file:
            for line in example:
                the_file.write(line + "\n")


def filename_to_docstring_example_file(filename, dirname):
    lines = get_lines(filename)
    examples = get_examples_docstring(lines)
    write_examples(examples, filename, dirname)


def filename_to_readme_example_file(dirname):
    lines = get_lines("README", filetype=".md")
    examples = get_examples_docstring(lines)
    write_examples(examples, "readme")


# filename_to_example_file("models")
os.makedirs("docstrings_examples", exist_ok=True)
filename_to_docstring_example_file("_utils", "docstrings")
filename_to_docstring_example_file("models", "docstrings")
filename_to_docstring_example_file("elbos", "docstrings")
filename_to_docstring_example_file("load", "docstrings")

filename_to_readme_example_file("docstrings")
