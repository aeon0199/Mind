from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepoHygieneTests(unittest.TestCase):
    def test_branchpoint_analyzer_default_allows_current_q1_slices(self):
        from scripts import analyze_branchpoints

        parser = analyze_branchpoints.build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.min_valid_pairs, 5)

    def test_ci_checks_active_runtime_package(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

        self.assertIn("python -m compileall -q src scripts", ci)

    def test_readme_quickstart_uses_unified_runtime_lab_cli(self):
        readme = (ROOT / "README.md").read_text()

        self.assertIn("python -m runtime_lab.cli.main observe", readme)
        self.assertIn("python -m runtime_lab.cli.main stress", readme)
        self.assertIn("python -m runtime_lab.cli.main control", readme)


if __name__ == "__main__":
    unittest.main()
