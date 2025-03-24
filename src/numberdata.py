import os
import numpy as np
import json


PERFECTS = [
    6, 28, 496, 8128, 33550336, 8589869056, 137438691328, 2305843008139952128,
    2658455991569831744654692615953842176,
    191561942608236107294793378084303638130997321548169216
    ]
M_PERFECTS = [
    1, 6, 28, 120, 496, 672, 8128, 30240, 32760, 523776, 2178540, 23569920,
    33550336, 45532800, 142990848, 459818240, 1379454720, 1476304896,
    8589869056, 14182439040, 31998395520, 43861478400, 51001180160,
    66433720320, 137438691328, 153003540480, 403031236608
    ]

PRIMITIVES = None
PSEUDOS = None
PRACTICALS = None
PRIM_PRACTICALS = None
PRIM_PRACTICALS_X = None

# Get absolute path of the project directory.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Functions for loading or saving data
def load_primitives():
    """Loads the 'primitive_pseudoperfect_list.txt' file content as an array
    """
    global PRIMITIVES
    global BASE_PATH
    if PRIMITIVES is None:
        data_path = os.path.join(
            BASE_PATH, "data", "primitive_pseudoperfect_list.txt"
        )  # go up one directory level then into data folder.

        with open(data_path, "r") as file:
            PRIMITIVES = np.array(file.read().split(", ")).astype(int)


def load_pseudos():
    """Loads the 'pseudoperfect_list.txt' file content as an array."""
    global PSEUDOS
    global BASE_PATH
    if PSEUDOS is None:
        data_path = os.path.join(
            BASE_PATH, "data", "pseudoperfect_list.txt"
        )  # go up one directory level then into data folder.

        with open(data_path, "r") as file:
            PSEUDOS = np.array(file.read().split(", ")).astype(int)


def load_practicals():
    """Loads the 'practical_list.txt' file content as an array."""
    global PRACTICALS
    if PRACTICALS is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, "data", "practical_list.txt")

        with open(data_path, "r") as file:
            PRACTICALS = np.array(file.read().split(", ")).astype(int)


def load_primpracticals():
    global PRIM_PRACTICALS
    if PRIM_PRACTICALS is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, "data", "primitive_practical_list.txt")

        with open(data_path, "r") as file:
            PRIM_PRACTICALS = np.array(file.read().split(", ")).astype(int)


def load_primpracticals_x():
    global PRIM_PRACTICALS_X
    if PRIM_PRACTICALS_X is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, "data", "primitive_practical_x_list.txt")

        with open(data_path, "r") as file:
            PRIM_PRACTICALS_X = np.array(
                file.read().split(", "),  dtype=np.int64)


def load_sequences():
    """Loads all the sequence data at once."""
    load_primitives()
    load_pseudos()
    load_practicals()
    load_primpracticals()
    load_primpracticals_x()


def load_entropies_data(data_type, filename="perfection_entropies"):
    """Load saved values from a file."""
    global BASE_PATH
    filepath = BASE_PATH + '\\data\\' + filename
    if data_type == "dict":
        if not os.path.exists(f"{filepath}.json"):
            print(f"File {filename}.json not found.")
            return None
        else:
            return load_dictionary_data(filepath)
    else:
        if not os.path.exists(f"{filepath}.txt"):
            print(f"File {filepath}.txt not found.")
            return None
        else:
            return load_list_data(filepath)


def load_dictionary_data(filepath):
    """Load saved key-value pairs from a line-by-line JSON file."""
    dictionary = {}
    last_complete_key = 'None found.'
    try:
        with open(f'{filepath}.json', 'r') as f:
            for line in f:
                try:
                    # Parse each line and update the dictionary
                    data = json.loads(line)
                    # Convert keys to integers and update the dictionary
                    for key, value in data.items():
                        dictionary[int(key)] = value
                    # Track the last successfully loaded key
                    last_complete_key = list(data.keys())[0]
                except json.JSONDecodeError:
                    # Print the last complete key when a decode error occurs
                    # this can happen when a previous write operation was
                    # interrupted
                    print(
                        f"Error loading data. Please, remove incomplete \
                    entries. Last complete key: {last_complete_key}"
                        )
                    return False
    except FileNotFoundError:
        return {}

    return dictionary


def load_list_data(filepath):
    """Load saved values from a .txt file."""
    perfection_entropies = []
    try:
        with open(f'{filepath}.txt', 'r') as f:
            for line in f:
                try:
                    n, entropy = line.split(", ")
                    perfection_entropies.append([n, entropy])
                except ValueError:
                    # Print the last complete value when a decode error occurs
                    print(
                        f"Error loading data. Please, remove incomplete \
entries. Last complete value: {n}"
                        )
                    return False
        return [(int(p[0]), float(p[1].replace("\n", ""))) for p in
                perfection_entropies]
    except FileNotFoundError:
        return []


def save_data(result, filename, data_type):
    """Saves generated values to a file."""
    global BASE_PATH
    filepath = BASE_PATH + "\\data\\" + filename
    if data_type == 'dict':
        return save_dictionary_data(result, filepath)
    else:
        return save_list_data(result, filepath)


def save_dictionary_data(result, filepath):
    """Saves generated key-value pair as a JSON object per line."""
    with open(f'{filepath}.json', 'a') as f:
        json.dump({result[0]: result[1]}, f)
        f.write('\n')  # Newline to separate each key-value pair


def save_list_data(result, filepath):
    """Saves generated values as a .txt file."""
    with open(f"{filepath}.txt", 'a') as file:
        file.write(f"{result[0]}, {result[1]}\n")


def check_filename(filename, ext):
    """Check if a file with the same name exists and prompt the user to
    overwrite or choose a new name."""
    global BASE_PATH
    filepath = BASE_PATH + "\\data\\" + filename
    if os.path.exists(f"{filepath}.{ext}"):
        confirm = input(
            f'A file with the name {filepath}.{ext} exists. Enter "o" to '
            'overwrite the file or "n" to choose a new name:'
        ).lower()
        if confirm in ["o", "n"]:
            if confirm == "o":
                print(f"Overwriting {filename}.{ext}")
                os.remove(f"{filepath}.{ext}")
                return filename
            else:
                filename = input("Choose a file name: ")
                return check_filename(filename, ext)
        else:
            print("Invalid input. Please try again.")
            filename = None
    return filename
