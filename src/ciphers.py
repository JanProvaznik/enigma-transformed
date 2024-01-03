import random
from typing import Callable, Optional
from enigma.machine import EnigmaMachine
from src.preprocessing import prepend_hello

rand = random.Random(42)


def generate_random_key(length: int) -> str:
    """Generates a random key of specified length."""
    return "".join(chr(rand.randint(ord("a"), ord("z"))) for _ in range(length))


def add_noise_to_text(text: str, noise_proportion: float) -> str:
    """Adds random noise to the text."""
    return "".join(
        c if rand.random() > noise_proportion else chr(rand.randint(ord("a"), ord("z")))
        for c in text
    )


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
    perm = rand.sample(range(26), 26)
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
    shifts: Optional[list[int]] = None
) -> Callable:  # -> ((text: str) -> str)
    """Creates a closure that shifts by the next shift in the list each time it's called

    Input: list of shifts
    Output: function that shifts by the next shift in the list each time it's called
    """
    if shifts is None:
        shifts = [3, 8, 14]
    i = 0

    def inner(text: str) -> str:
        nonlocal i
        shifted = caesar(text, shifts[i])
        i = (i + 1) % len(shifts)
        return shifted

    return inner


def make_const_enigma(
    ring_setting: Optional[list[int]] = None,
    display: str = "WXC",
    noise_proportion: Optional[float] = None,
) -> Callable:
    """Creates a closure that encrypts text using an enigma machine with the same settings each time it's called

    Input: enigma machine ring settings, display, noise proportion
    Output: function that shifts by the next shift in the list each time it's called

    WARNING: non-letter characters are converted to 'X'
    """
    if ring_setting is None:
        ring_setting = [1, 20, 11]

    machine = EnigmaMachine.from_key_sheet(
        rotors="II IV V",
        reflector="B",
        ring_settings=ring_setting,
        # plugboard_settings='AV BS CG DL FU HZ IN KM OW' # let's not use this for simplicity
    )

    def inner(text: str) -> str:
        nonlocal machine
        nonlocal display
        machine.set_display(display)
        if noise_proportion:
            text = add_noise_to_text(text, noise_proportion)
        shifted = f"{machine.process_text(text)}"
        return shifted.lower()  # the process_text function returns uppercase

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


def make_multi_vignere(keys: Optional[list[str]] = None) -> Callable:
    """Creates a closure that applies vignere with a key and changes the key with each application
    Input: list of keys
    Output: function that can be used to roll through the keys and apply vignere
    """
    if keys is None:
        keys = ["bm", "xp", "ek"]
    i = 0

    def inner(text: str) -> str:
        nonlocal i
        shifted = vignere(text, keys[i])
        i = (i + 1) % len(keys)
        return shifted

    return inner


def random_vignere(text: str, key_length: int) -> str:
    """Vigenère cipher with a random key of specified length."""
    key = generate_random_key(key_length)
    return vignere(text, key)


def noisy_random_vignere(
    text: str, key_length: int, noise_proportion: float = 0.15
) -> str:
    """Vigenère cipher with a random key of specified length and noise."""
    noisy_text = add_noise_to_text(text, noise_proportion)
    return random_vignere(noisy_text, key_length)


def random_vignere2(text: str) -> str:
    """Vigenère cipher with a random 2-letter key."""
    return random_vignere(text, 2)


def random_vignere3(text: str) -> str:
    """Vigenère cipher with a random 3-letter key."""
    return random_vignere(text, 3)


def noisy_random_vignere2(text: str, noise_proportion: float = 0.15) -> str:
    """Vigenère cipher with a random 2-letter key and noise."""
    return noisy_random_vignere(text, 2, noise_proportion)


def noisy_random_vignere3(text: str, noise_proportion: float = 0.15) -> str:
    """Vigenère cipher with a random 3-letter key and noise."""
    return noisy_random_vignere(text, 3, noise_proportion)
