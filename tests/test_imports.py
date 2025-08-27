def test_imports():
    import importlib
    importlib.import_module("src.main")

def test_libs_available():
    import pandas, sklearn
    assert pandas.__version__
    assert sklearn.__version__
