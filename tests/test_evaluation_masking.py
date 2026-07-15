import unittest

import torch

from maia2.main import evaluate


class FixedLogitsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, boards, elos_self, elos_oppo):
        batch_size = boards.shape[0]
        logits = torch.tensor([-5.0, -3.0, -4.0], device=boards.device).expand(
            batch_size, -1
        )
        logits = logits + self.anchor
        side_info = torch.zeros((batch_size, 1), device=boards.device)
        value = torch.zeros(batch_size, device=boards.device)
        return logits, side_info, value


class EvaluationMaskingRegressionTest(unittest.TestCase):
    def test_illegal_move_cannot_win_argmax_when_legal_logits_are_negative(self):
        batch = (
            torch.zeros((1, 18, 8, 8)),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[1, 1, 0]]),
            torch.zeros((1, 1)),
        )

        correct, count = evaluate(FixedLogitsModel(), [batch])

        self.assertEqual(correct, 1)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
