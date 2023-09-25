import sys
sys.path.append(".")

from src.ciphers import *

def test_nothing():
    assert nothing("hello") == "hello"
    assert nothing("hello hello") == "hello hello"
    assert nothing("hello.hello") == "hello.hello"
    assert nothing("") == ""

def test_caesar():
    assert caesar("abc", 3) == "def"
    assert caesar("xyz", 3) == "abc"
    assert caesar("abc", 5) == "fgh"
    assert caesar("abc abc", 5) == "fgh fgh"
    assert caesar("abc.abc", 5) == "fgh.fgh"

def test_make_multi_caesar():
    cipher = make_multi_caesar([3, 8, 14])
    assert cipher("abc") == "def"
    assert cipher("abc") == "ijk"
    assert cipher("abc") == "opq"
    assert cipher("abc") == "def"
    assert cipher("abc abc") == "ijk ijk"
    assert cipher("abc.abc") == "opq.opq"

def test_caesar_random_hint():
    cipher = caesar_random_hello_hint("hello world")
    assert len(cipher) == len("hello world") + 6

def test_caesar_random():
    cipher = caesar_random("hello world")
    assert len(cipher) == len("hello world")
    assert cipher[5] == " "
    c2 = caesar_random("hello.world")
    assert len(c2) == len("hello.world")
    assert c2[5] == "."

#TODO: fix enigma
# def test_enigma_encrypt_all_the_same():
#     cipher = enigma_encrypt_all_the_same("hello world")
#     assert len(cipher) == len("hello world")
#     assert cipher != "hello world"
#     cipher2 = enigma_encrypt_all_the_same("hello world")
#     assert cipher[5] == " "
#     assert cipher2 == cipher

def test_substitute():
    mapping = {"a": "b", "b": "c", "c": "a"}
    assert substitute("abc", mapping) == "bca"

def test_create_substitution_dict():
    perm = ["b", "c", "a", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    mapping = create_substitution_dict(perm)
    assert mapping["a"] == "b"
    assert mapping["b"] == "c"
    assert mapping["c"] == "a"
    assert mapping["d"] == "d"
    assert mapping[" "] == " "

def test_random_substitution():
    cipher, mapping = random_substitution("hello world")
    assert len(cipher) == len("hello world")
    assert cipher != "hello world" # this can misfire, but it's unlikely
    assert len(mapping) == 27
    assert set(mapping.keys()) == set("abcdefghijklmnopqrstuvwxyz ")
    assert set(mapping.values()) == set("abcdefghijklmnopqrstuvwxyz ")