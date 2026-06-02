import torch

from styletransfer.features import gram_matrix


def test_gram_matrix_shape() -> None:
    tensor = torch.ones((1, 3, 4, 4))

    gram = gram_matrix(tensor)

    assert gram.shape == (3, 3)


def test_gram_matrix_is_normalized() -> None:
    tensor = torch.ones((1, 2, 2, 2))

    gram = gram_matrix(tensor)

    expected = torch.full((2, 2), 0.5)
    torch.testing.assert_close(gram, expected)
