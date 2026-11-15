"""Mathematical and leakage checks for diffusion evidence, using CPU toy graphs."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graphmine.evidence import diffusion_evidence, FusionGate
from graphmine.prop import gcn_adj, propagate


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Two labelled leaves -> centre; separate one-label component; isolated node.
        edge = torch.tensor([[0, 2, 1, 2, 3, 4], [2, 0, 2, 1, 4, 3]])
        self.adj = gcn_adj(edge, 6)
        self.y = torch.tensor([0, 0, 1, 1, 0, 1])
        self.sources = torch.tensor([0, 1, 3])

    def test_equal_entropy_different_source_count(self):
        q, s = diffusion_evidence(self.adj, self.y, self.sources, 2)
        self.assertAlmostEqual(q[2, 0].item(), 1)
        self.assertAlmostEqual(q[4, 1].item(), 1)
        self.assertAlmostEqual(s[2, 1].item(), 2, places=5)
        self.assertAlmostEqual(s[4, 1].item(), 1, places=5)
        self.assertEqual(s[5, 2].item(), 0)
        self.assertTrue(torch.equal(q[5], torch.tensor([.5, .5])))

    def test_no_heldout_label_access(self):
        q, s = diffusion_evidence(self.adj, self.y, self.sources, 2)
        changed = self.y.clone()
        changed[torch.tensor([2, 4, 5])] = -987  # invalid labels outside source mask
        q2, s2 = diffusion_evidence(self.adj, changed, self.sources, 2)
        self.assertTrue(torch.equal(q, q2))
        self.assertTrue(torch.equal(s, s2))

    def test_exact_operator_and_sketch(self):
        _, s = diffusion_evidence(self.adj, self.y, self.sources, 2)
        initial = torch.zeros(6, 3)
        initial[self.sources, torch.arange(3)] = 1
        h = propagate(self.adj, initial, .9, 50)
        exact = h.sum(1).square() / h.square().sum(1).clamp_min(1e-24)
        torch.testing.assert_close(s[:, 1], exact)
        _, approx = diffusion_evidence(self.adj, self.y, self.sources, 2,
                                       probes=8192, exact_limit=0, seed=7)
        torch.testing.assert_close(approx[:, 1], exact, atol=.07, rtol=.04)

    def test_router_abstention(self):
        p = torch.tensor([[[.2, .8]], [[.9, .1]]])
        gate = FusionGate(2, 0, torch.zeros(2), 1)
        q, w = gate(p, torch.empty(1, 2, 0), torch.tensor([False]))
        torch.testing.assert_close(q, p[0])
        self.assertEqual(w[0, 1].item(), 0)

    def test_class_aware_zero_gate_preserves_stacking(self):
        from graphmine.ensemble import cooperate
        torch.manual_seed(0)
        p = torch.randn(3, 8, 4).softmax(2)
        theta = torch.randn(3, 4, 4)
        gate = FusionGate(3, 5, torch.zeros(3), 1, theta)
        # Include missing LP evidence: the static class-prior contribution must
        # remain unchanged at zero local adaptation too.
        q, _ = gate(p, torch.randn(8, 3, 5), torch.arange(8) % 2 == 0)
        torch.testing.assert_close(q, cooperate(p, theta).softmax(1))
        logq, _ = gate(p, torch.randn(8, 3, 5), torch.ones(8, dtype=torch.bool), log_output=True)
        torch.testing.assert_close(logq.exp(), q)
        torch.nn.functional.nll_loss(logq, torch.arange(8) % 4).backward()
        self.assertGreater(gate.weight.grad.norm().item(), 0)
        self.assertIsNone(gate.theta.grad)


if __name__ == '__main__':
    unittest.main()
