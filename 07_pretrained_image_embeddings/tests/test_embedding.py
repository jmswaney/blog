import embedding

def test_img_size():
    assert embedding.main("tests/dog.jpg") == (333, 516, 3)
