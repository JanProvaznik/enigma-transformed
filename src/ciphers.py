import random
import numpy as np
from enigma.machine import EnigmaMachine
rand = random.Random(42)

def substitute(text, mapping):
    """
    mapping is a dict that for each letter in the alphabet
    maps to another letter in the alphabet
    
    enusre the map is a bijection!
    """
    return ''.join(mapping.get(c, c) for c in text)

def create_substitution_dict(permutation):
    """
    permutation is a list of letters in the alphabet
    """
    d = {chr(ord('A') + i): permutation[i] for i in range(len(permutation))}
    d[' '] = ' '
    return d

# this should be impossible to crack without frequence analysis
def random_substitution(text):
    """
    substitution cipher should be invertible
    """
    perm = np.random.permutation(26)
    perm_str = [chr(ord('A') + i) for i in perm]
    mapping = create_substitution_dict(perm_str)
    return (substitute(text, mapping), mapping)

def nothing(text):
    return text

# Caesar
def caesar(text, shift=3):
    return ''.join(chr((ord(c) - ord('A') + shift) % 26 + ord('A')) for c in text)

def caesar_random_hint(text):
    shift = rand.randint(0, 25)
    # convert shift to char
    shift_char = chr(ord('A') + shift)
    return f"{shift_char}{caesar(text, shift)}"

def caesar_random(text):
    shift = rand.randint(0, 25)
    return caesar(text, shift)

# Enigma
machine = None
def enigma_encrypt_all_the_same(text, start_display='ABC', ring_settings=[0,0,0]):
    if machine == None:
       machine = EnigmaMachine.from_key_sheet(
       rotors='I II III',
       reflector='B',
       ring_settings=ring_settings,
       plugboard_settings=None)
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"

def enigma_encrypt_random(text, seed=42):
    machine = EnigmaMachine.from_key_sheet(
        rotors='I II III',
        reflector='B',
        ring_settings=[0,0,0],
        plugboard_settings=None)
    start_display = ''.join(rand.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(3))
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"