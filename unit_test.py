# unit_test.py

def test_basic_math():
    """A trivial test to check that tests run."""
    assert 1 + 1 == 2

def test_label_mapping_exists():
    """Check that labelIndex2Name function exists."""
    from app import labelIndex2Name
    assert callable(labelIndex2Name)
