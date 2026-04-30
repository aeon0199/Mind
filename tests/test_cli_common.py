import unittest

from runtime_lab.cli._common import parse_seeds, resolve_probe_layers, resolve_semantic_layer


class CliCommonTests(unittest.TestCase):
    def test_parse_seed_range(self):
        self.assertEqual(parse_seeds("0-3"), [0, 1, 2, 3])

    def test_parse_seed_list(self):
        self.assertEqual(parse_seeds("2, 4,6"), [2, 4, 6])

    def test_auto_probe_layers_resolve_across_depth(self):
        self.assertEqual(resolve_probe_layers("auto", 28), [6, 13, 20, 27])

    def test_semantic_late_layer_resolves_to_last_layer(self):
        self.assertEqual(resolve_semantic_layer("late", 28), 27)


if __name__ == "__main__":
    unittest.main()
