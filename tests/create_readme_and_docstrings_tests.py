import ast
import os


dir_docstrings = "docstrings_examples"
dir_readme = "readme_examples"


def get_lines(path_to_file, filename, filetype=".py"):
    with open(f"{path_to_file}{filename}{filetype}") as file:
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


def get_example_readme(lines):
    example = []
    in_example = False
    for line in lines:
        line = line.lstrip()
        if len(line) > 2:
            if line[0:3] == "```":
                if in_example is False:
                    in_example = True
                else:
                    in_example = False
            elif in_example is True:
                example.append(line)
    example.pop(0)  # The first is pip install pyPLNmodels which is not python code.
    return [example]


def write_examples(examples, filename):
    for i in range(len(examples)):
        example = examples[i]
        nb_example = str(i + 1)
        example_filename = f"test_{filename}_example_{nb_example}.py"
        try:
            os.remove(example_filename)
        except FileNotFoundError:
            pass
        with open(example_filename, "a") as the_file:
            for line in example:
                the_file.write(line + "\n")


def filename_to_docstring_example_file(filename, dirname):
    lines = get_lines("../pyPLNmodels/", filename)
    examples = get_examples_docstring(lines)
    write_examples(examples, filename)


def filename_to_readme_example_file():
    lines = get_lines("../", "README", filetype=".md")
    examples = get_example_readme(lines)
    write_examples(examples, "readme")


os.makedirs(dir_readme, exist_ok=True)
filename_to_readme_example_file()

os.makedirs("docstrings_examples", exist_ok=True)
filename_to_docstring_example_file("_utils", dir_docstrings)
filename_to_docstring_example_file("models", dir_docstrings)
filename_to_docstring_example_file("elbos", dir_docstrings)
filename_to_docstring_example_file("load", dir_docstrings)
