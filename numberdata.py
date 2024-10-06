import os
import numpy as np


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

# Global variables to store file contents
PRIMITIVES = None
PSEUDOS = None
PRACTICALS = None
PRIM_PRACTICALS = None
PRIM_PRACTICALS_X = None


def load_primitives():
    """Loads the 'primitive_pseudoperfect_list.txt' file content as an array
    """
    global PRIMITIVES
    if PRIMITIVES is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, 'primitive_pseudoperfect_list.txt')

        with open(data_path, "r") as file:
            PRIMITIVES = np.array(file.read().split(", ")).astype(int)


def load_pseudos():
    """Loads the 'pseudoperfect_list.txt' file content as an array."""
    global PSEUDOS
    if PSEUDOS is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, 'pseudoperfect_list.txt')

        with open(data_path, "r") as file:
            PSEUDOS = np.array(file.read().split(", ")).astype(int)


def load_practicals():
    """Loads the 'practical_list.txt' file content as an array."""
    global PRACTICALS
    if PRACTICALS is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, 'practical_list.txt')

        with open(data_path, "r") as file:
            PRACTICALS = np.array(file.read().split(", ")).astype(int)


def load_primpracticals():
    global PRIM_PRACTICALS
    if PRIM_PRACTICALS is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, 'primitive_practical_list.txt')

        with open(data_path, "r") as file:
            PRIM_PRACTICALS = np.array(file.read().split(", ")).astype(int)


def load_primpracticals_x():
    global PRIM_PRACTICALS_X
    if PRIM_PRACTICALS_X is None:
        base_path = os.path.dirname(__file__)  # Get the current module path
        data_path = os.path.join(base_path, 'primitive_practical_x_list.txt')

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
