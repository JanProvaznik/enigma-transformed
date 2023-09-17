import random
import numpy as np
from enigma.machine import EnigmaMachine

rand = random.Random(42)


def substitute(text: str, mapping: dict[str, str]):
    """
    mapping is a dict that for each letter in the alphabet
    maps to another letter in the alphabet

    enusre the map is a bijection!
    """
    return "".join(mapping.get(c, c) for c in text)


def create_substitution_dict(permutation: list[str]) -> dict[str, str]:
    """
    permutation is a list of letters in the alphabet
    """
    d = {chr(ord("A") + i): permutation[i] for i in range(len(permutation))}
    d[" "] = " "
    return d


# this should be impossible to crack without frequence analysis on large dataset
def random_substitution(text: str) -> tuple[str, dict[str, str]]:
    """
    substitution cipher should be invertible
    """
    perm = np.random.permutation(26)
    perm_str = [chr(ord("A") + i) for i in perm]
    mapping = create_substitution_dict(perm_str)
    return (substitute(text, mapping), mapping)


def nothing(text: str) -> str:
    return text


# Caesar
def caesar(text: str, shift: int = 3) -> str:
    return "".join(
        chr((ord(c) - ord("A") + shift) % 26 + ord("A")) if c.isalpha() else c
        for c in text
    )

def make_multi_caesar(shifts: list[int] = [3, 8, 14]): # -> ((text: str) -> str)
    i = 0
    def inner(text: str) -> str:
        nonlocal i
        shifted = caesar(text, shifts[i])
        i = (i + 1) % len(shifts)
        return shifted
    return inner


def caesar_random_hint(text: str) -> str:
    shift = rand.randint(0, 25)
    # convert shift to char
    shift_char = chr(ord("A") + shift)
    return f"{shift_char} {caesar(text, shift)}"


def caesar_random(text: str) -> str:
    shift = rand.randint(0, 25)
    return caesar(text, shift)


machine = None


def enigma_encrypt_all_the_same(
    text: str, start_display: str = "ABC", ring_settings: list[int] = [0, 0, 0]
) -> str:
    if machine == None:
        machine = EnigmaMachine.from_key_sheet(
            rotors="I II III",
            reflector="B",
            ring_settings=ring_settings,
            plugboard_settings=None,
        )
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"
