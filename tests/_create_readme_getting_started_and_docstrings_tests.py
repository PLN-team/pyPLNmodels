import os
import glob

OUTPUT_DIR_DOCSTRINGS = "docstrings_examples"
OUTPUT_DIR_README = "readme_examples"
OUTPUT_DIR_GETTING_STARTED = "getting_started"

DIR_LOAD_DATA = "load_data"
DIR_SAMPLING = "sampling"
DIR_MODELS = "models"
DIR_UTILS = "utils"


def _get_lines(path_to_file, filename, filetype=".py"):
    with open(f"{path_to_file}{filename}{filetype}", encoding="utf-8") as file:
        lines = []
        for file_line in file:
            rstrip_line = file_line.rstrip()
            if len(rstrip_line) > 4:
                if rstrip_line[0:3] != "pip":
                    lines.append(rstrip_line)
            else:
                lines.append(rstrip_line)
    return lines


def _get_examples_docstring(lines):
    examples = []
    in_example = False
    example = []
    for doc_line in lines:
        doc_line = doc_line.lstrip()
        if len(doc_line) > 3:
            if doc_line[0:3] == ">>>":
                in_example = True
                example.append(doc_line[4:])
            else:
                if in_example is True:
                    examples.append(example)
                    example = []
                in_example = False
    return examples


def _get_example_readme(lines):
    example = []
    in_example = False
    for readme_line in lines:
        if readme_line.lstrip().startswith("```"):
            if in_example:
                in_example = False
            else:
                if "not run" not in readme_line:
                    in_example = True
        elif in_example:
            example.append(readme_line)
    if example and example[-1].strip() == "```":
        example.pop()
    return [example]


def _write_file(examples, filename, string_definer, directory):
    for i, example in enumerate(examples):
        nb_example = str(i + 1)
        example_filename = (
            f"{directory}/test_{filename}_{string_definer}_{nb_example}.py"
        )
        try:
            os.remove(example_filename)
        except FileNotFoundError:
            pass
        with open(example_filename, "a", encoding="utf-8") as the_file:
            for example_line in example:
                the_file.write(example_line + "\n")


def _filename_to_readme_example_file():
    lines = _get_lines("../", "README", filetype=".md")
    examples = _get_example_readme(lines)
    _write_file(examples, "readme", "example", directory=OUTPUT_DIR_README)


os.makedirs(OUTPUT_DIR_README, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCSTRINGS, exist_ok=True)
os.makedirs(OUTPUT_DIR_GETTING_STARTED, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCSTRINGS + "/" + DIR_LOAD_DATA, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCSTRINGS + "/" + DIR_SAMPLING, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCSTRINGS + "/" + DIR_MODELS, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCSTRINGS + "/" + DIR_UTILS, exist_ok=True)


def _find_all_files(directory):
    py_files = glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)
    py_files_relative = [
        os.path.splitext(os.path.relpath(f, directory))[0] for f in py_files
    ]
    return py_files_relative


def _get_last_directory_name(path):
    return os.path.split(path)[-1]


def _find_all_directories(path):
    _directories = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            _directories.append(item_path)
    return _directories


def _filename_to_docstring_example_file(filename, dirname, path_to_file):
    lines = _get_lines(path_to_file + "/", filename)
    examples = _get_examples_docstring(lines)
    dirname += "/" + _get_last_directory_name(path_to_file)
    _write_file(examples, filename, "example", directory=dirname)


directories = _find_all_directories("../pyPLNmodels")
for _directory in directories:
    files = _find_all_files(_directory)
    for _file in files:
        if os.path.isfile(_directory + "/" + _file + ".py"):
            _filename_to_docstring_example_file(
                _file, OUTPUT_DIR_DOCSTRINGS, _directory
            )
_filename_to_readme_example_file()

new_lines = []
lines_getting_started = _get_lines("./", "untestable_getting_started")
for line in lines_getting_started:
    if len(line) > 20:
        if line[0:11] != "get_ipython":
            new_lines.append(line)
    else:
        new_lines.append(line)
_write_file([new_lines], "getting_started", "", OUTPUT_DIR_GETTING_STARTED)
