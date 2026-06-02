from PIL import Image

from styletransfer.image_io import image_to_tensor, tensor_to_image


def test_image_round_trip_keeps_rgb_mode() -> None:
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))

    tensor = image_to_tensor(image, target_size=32)
    restored = tensor_to_image(tensor)

    assert tensor.shape == (1, 3, 32, 32)
    assert restored.mode == "RGB"
    assert restored.size == (32, 32)
