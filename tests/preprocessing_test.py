import sys
sys.path.append(".")
from src.preprocessing import *
from tempfile import NamedTemporaryFile
import os

def test_generate_random_dataset():
    dataset = generate_random_dataset(rows=10, min_length=5, max_length=10, space_frequency=0.2, seed=42)
    assert len(dataset) == 10, "Row count mismatch"
    for row in dataset:
        assert 5 <= len(row) <= 10, "Row length out of bounds"

def test_load_dataset():
    # Creating a temp file with 20 rows for testing
    with NamedTemporaryFile(mode='w+', delete=False) as f:
        f.writelines([f"Line {i}\n" for i in range(20)])
        temp_file_path = f.name

    try:
        dataset = load_dataset(rows=10, min_length=5, max_length=10, file_path=temp_file_path, seed=42)
        assert len(dataset) == 10, "Row count mismatch"
        for row in dataset:
            assert 5 <= len(row) <= 10, "Row length out of bounds"
    finally:
        os.remove(temp_file_path)

def test_only_letters():
    assert only_letters("123 ABC !@#") == "abc", "Special chars not removed"
    assert only_letters("123 ABC !@#", preserve_spaces=True) == " abc ", "Spaces not preserved"

def test_replace_digits():
    assert replace_digits("123") == "onetwothree", "Digits not replaced correctly"

def test_preprocess_text():
    assert preprocess_text("123 ABC !@#") == " abc ", "Preprocessing failed"
    assert preprocess_text("123 ABC !@#", convert_digits=True) == "onetwothree abc ", "Digits not converted"

def test_prepend_hello():
    assert prepend_hello("world") == "hello world", "'hello' not prepended correctly"
