import ast
import os


def get_lines(filename):
    with open(f"../pyPLNmodels/{filename}.py") as file:
        lines = [line.rstrip() for line in file]
    return lines


def get_examples(lines):
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


def write_examples(examples, prefix_filename):
    for i in range(len(examples)):
        example = examples[i]
        nb_example = str(i + 1)
        example_filename = f"examples/{prefix_filename}_example_{nb_example}.py"
        try:
            os.remove(example_filename)
        except FileNotFoundError:
            pass
        with open(example_filename, "a") as the_file:
            for line in example:
                the_file.write(line + "\n")


def filename_to_example_file(filename):
    lines = get_lines(filename)
    examples = get_examples(lines)
    write_examples(examples, filename)


# filename_to_example_file("models")
os.makedirs("examples", exist_ok=True)
filename_to_example_file("_utils")
filename_to_example_file("models")
filename_to_example_file("elbos")
filename_to_example_file("load")
