def test_import():
    import backend_main
    assert backend_main.app is not None
    print("✅ API import OK")

def test_templates():
    import backend_main
    assert hasattr(backend_main, 'app')
    print("✅ App loaded")
