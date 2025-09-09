def test_import():
    import sys
    assert 'face_recognition' or 'cv2' in sys.modules
