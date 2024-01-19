import ast
import os


dir_docstrings = "docstrings_examples"
dir_readme = "readme_examples"
dir_getting_started = "getting_started"


def get_lines(path_to_file, filename, filetype=".py"):
    with open(f"{path_to_file}{filename}{filetype}") as file:
        lines = []
        for line in file:
            rstrip_line = line.rstrip()
            if len(rstrip_line) > 4:
                if rstrip_line[0:3] != "pip":
                    lines.append(rstrip_line)
            else:
                lines.append(rstrip_line)
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
    example.pop()  # The last line is pip install pyPLNmodels which is not python code.
    return [example]


def write_file(examples, filename, string_definer, dir):
    for i in range(len(examples)):
        example = examples[i]
        nb_example = str(i + 1)
        example_filename = f"{dir}/test_{filename}_{string_definer}_{nb_example}.py"
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
    write_file(examples, filename, "example", dir=dirname)


def filename_to_readme_example_file():
    lines = get_lines("../", "README", filetype=".md")
    examples = get_example_readme(lines)
    write_file(examples, "readme", "example", dir=dir_readme)


lines_getting_started = get_lines("./", "test_getting_started")
new_lines = []
for line in lines_getting_started:
    if len(line) > 20:
        if line[0:11] != "get_ipython":
            new_lines.append(line)
    else:
        new_lines.append(line)


os.makedirs(dir_readme, exist_ok=True)
os.makedirs(dir_docstrings, exist_ok=True)
os.makedirs(dir_getting_started, exist_ok=True)

write_file([new_lines], "getting_started", "", dir_getting_started)

filename_to_readme_example_file()


filename_to_docstring_example_file("_utils", dir_docstrings)
filename_to_docstring_example_file("models", dir_docstrings)
filename_to_docstring_example_file("elbos", dir_docstrings)
filename_to_docstring_example_file("load", dir_docstrings)
