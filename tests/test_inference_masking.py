import unittest

import torch

from maia2 import inference
from maia2.utils import create_elo_dict, get_all_possible_moves


NORMAL_FEN = "rn1q1rk1/ppp2ppp/4bn2/3p3P/4p3/P3P3/1PPPBPPb/RNBQK3 w Q - 0 11"
TERMINAL_FEN = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"


class DummyModel(torch.nn.Module):
    def __init__(self, move_count):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.move_count = move_count

    def forward(self, boards, elos_self, elos_oppo):
        batch_size = boards.shape[0]
        logits = torch.linspace(-1, 1, self.move_count, device=boards.device)
        logits = logits.unsqueeze(0).repeat(batch_size, 1) + self.anchor
        value = torch.zeros(batch_size, device=boards.device) + self.anchor
        return logits, None, value


class MaskingRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moves = get_all_possible_moves()
        cls.move_to_index = {move: i for i, move in enumerate(cls.moves)}
        cls.index_to_move = {i: move for move, i in cls.move_to_index.items()}
        cls.elo_dict = create_elo_dict()
        cls.model = DummyModel(len(cls.moves))

    def test_masked_softmax_zeros_illegal_moves_and_normalizes_each_row(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        legal_moves = torch.tensor([[1, 0, 1], [0, 1, 1]])

        probs = inference._masked_softmax(logits, legal_moves)

        self.assertEqual(probs[0, 1].item(), 0.0)
        self.assertEqual(probs[1, 0].item(), 0.0)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))

    def test_masked_softmax_rejects_only_the_empty_batch_rows(self):
        logits = torch.zeros((2, 3))
        legal_moves = torch.tensor([[1, 0, 1], [0, 0, 0]])

        with self.assertRaisesRegex(ValueError, r"batch rows: \[1\]"):
            inference._masked_softmax(logits, legal_moves)

    def test_terminal_position_has_a_clear_error(self):
        prepared = [self.move_to_index, self.elo_dict, self.index_to_move]

        with self.assertRaisesRegex(ValueError, "position without legal moves"):
            inference.inference_each(
                self.model, prepared, TERMINAL_FEN, 1500, 1498
            )

    def test_normal_position_probabilities_sum_to_one(self):
        prepared = [self.move_to_index, self.elo_dict, self.index_to_move]

        move_probs, _ = inference.inference_each(
            self.model, prepared, NORMAL_FEN, 1500, 1498
        )

        self.assertAlmostEqual(sum(move_probs.values()), 1.0, delta=0.005)


if __name__ == "__main__":
    unittest.main()
