import random
from enigma.machine import EnigmaMachine
rand = random.Random(42)
def caesar(text, shift=3):
    return ''.join(chr((ord(c) - ord('A') + shift) % 26 + ord('A')) for c in text)

def caesar_random_hint(text):
    shift = rand.randint(0, 25)
    return f"{shift}{caesar(shift)(text)}"

def caesar_random(text):
    shift = rand.randint(0, 25)
    return caesar(text, shift)

def nothing(text):
    return text

# TODO don't create a new machine for each sentence
def encrypt_all_the_same(text):
    machine = EnigmaMachine.from_key_sheet(
       rotors='I II III',
       reflector='B',
       ring_settings=[0, 0, 0],
       plugboard_settings=None)
    start_display = 'ABC'
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"

# TODO don't create a new machine for sentence and use randomness correctly
def encrypt_random(text, seed=42):
    machine = EnigmaMachine.from_key_sheet(
        rotors='I II III',
        reflector='B',
        ring_settings=[0,0,0],
        plugboard_settings=None)
    start_display = ''.join(rand.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(3))
    machine.set_display(start_display)
    return f"{start_display}{machine.process_text(text)}"