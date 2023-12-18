import random
from typing import Callable
from enigma.machine import EnigmaMachine
from src.preprocessing import prepend_hello

rand = random.Random(42)


# arbitrary substitution cipher
def substitute(text: str, mapping: dict[str, str]):
    """
    mapping is a dict that for each letter in the alphabet
    maps to another letter in the alphabet

    enusre the map is a bijection!
    """
    return "".join(mapping.get(c, c) for c in text)


def create_substitution_dict(permutation: list[str]) -> dict[str, str]:
    """creates a substitution dict from a permutation of the alphabet"""
    d = {chr(ord("a") + i): permutation[i] for i in range(len(permutation))}
    d[" "] = " "
    return d


def random_substitution(text: str) -> tuple[str, dict[str, str]]:
    """
    creates a random substitution cipher and applies it to the text
    Input: text
    Output: tuple of (text, mapping)
    """
    perm = random.sample(range(26), 26)
    perm_str = [chr(ord("a") + i) for i in perm]
    mapping = create_substitution_dict(perm_str)
    return (substitute(text, mapping), mapping)


def nothing(text: str) -> str:
    """identity function"""
    return text


# Caesar
def caesar(text: str, shift: int = 3) -> str:
    """Caesar cipher that ignores non-alphabetic characters
    Input: text, shift
    Output: text shifted by shift
    """
    return "".join(
        chr((ord(c) - ord("a") + shift) % 26 + ord("a")) if c.isalpha() else c
        for c in text
    )


def make_multi_caesar(
    shifts: list[int] = [3, 8, 14]
) -> Callable:  # -> ((text: str) -> str)
    """Creates a closure that shifts by the next shift in the list each time it's called

    Input: list of shifts
    Output: function that shifts by the next shift in the list each time it's called
    """
    i = 0

    def inner(text: str) -> str:
        nonlocal i
        shifted = caesar(text, shifts[i])
        i = (i + 1) % len(shifts)
        return shifted

    return inner


def make_const_enigma(ring_setting = [3, 22, 12], display="WXC") -> Callable:
    """Creates a closure that encrypts text using an enigma machine with the same settings each time it's called
    
    Input: enigma machine ring settings, display
    Output: function that shifts by the next shift in the list each time it's called

    WARNING: non-letter characters are converted to 'X'
    """

    machine = EnigmaMachine.from_key_sheet(
        rotors="II IV V",
        reflector="B",
        ring_settings=[1, 20, 11],
        # plugboard_settings='AV BS CG DL FU HZ IN KM OW' # let's not use this for simplicity
    )

    def inner(text: str) -> str:
        nonlocal machine
        nonlocal display
        machine.set_display(display)
        shifted = f"{machine.process_text(text)}"
        return shifted.lower() # the process_text function returns uppercase

    return inner


def caesar_random_hello_hint(text: str) -> str:
    """Caesar cipher that prepends 'hello' to the text and then shifts by a random amount"""
    shift = rand.randint(0, 25)
    return caesar(prepend_hello(text), shift)


def caesar_random(text: str) -> str:
    """Caesar cipher that shifts by a random amount"""
    shift = rand.randint(0, 25)
    return caesar(text, shift)


# Vignere
def vignere(text: str, key: str) -> str:
    """Vignere cipher that ignores non-alphabetic characters
    Input: text, key
    Output: text with Vignere algorithm applied with key

    rolling key: output[i] = (key[i % len(key)] + text[i]) % 26
    nonalphabetic characters are preserved but the key still gets rolled

    """
    key_len = len(key)
    if key_len == 0:
        return text

    encrypted = []

    for idx, c in enumerate(text):
        if c.isalpha():
            shift = ord(key[idx % key_len]) - ord("a")
            encrypted.append(caesar(c, shift))
        else:
            encrypted.append(c)

    return "".join(encrypted)


def vignere2(text: str, key: str = "cd") -> str:
    """Vignere cipher with a 2-letter key"""
    assert len(key) == 2
    return vignere(text, key)


def vignere3(text: str, key: str = "cmx") -> str:
    """Vignere cipher with a 3-letter key"""
    assert len(key) == 3
    return vignere(text, key)


def make_multi_vignere(keys: list[str] = ["bm", "xp", "ek"]) -> Callable:
    """Creates a closure that applies vignere with a key and changes the key with each application
    Input: list of keys
    Output: function that can be used to roll through the keys and apply vignere
    """
    i = 0

    def inner(text: str) -> str:
        nonlocal i
        shifted = vignere(text, keys[i])
        i = (i + 1) % len(keys)
        return shifted

    return inner


def random_vignere2(text: str) -> str:
    """Vignere cipher with a random 2-letter key"""
    k1 = chr(rand.randint(ord("a"), ord("z")))
    k2 = chr(rand.randint(ord("a"), ord("z")))
    return vignere2(text, f"{k1}{k2}")
