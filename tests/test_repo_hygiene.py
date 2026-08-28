from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepoHygieneTests(unittest.TestCase):
    def test_branchpoint_analyzer_defaults_to_repeated_run_grouped_evaluation(self):
        from scripts import analyze_branchpoints

        parser = analyze_branchpoints.build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.repeats, 20)
        self.assertEqual(args.min_runs, 3)

    def test_ci_checks_active_runtime_package(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

        self.assertIn("python -m compileall -q src analysis experiments tools", ci)

    def test_readme_quickstart_uses_unified_runtime_lab_cli(self):
        readme = (ROOT / "README.md").read_text()

        self.assertIn("python -m runtime_lab.cli.main observe", readme)
        self.assertIn("python -m runtime_lab.cli.main stress", readme)
        self.assertIn("python -m runtime_lab.cli.main control", readme)


if __name__ == "__main__":
    unittest.main()


    def test_repository_layout_separates_active_and_archived_surfaces(self):
        self.assertTrue((ROOT / "analysis").is_dir())
        self.assertTrue((ROOT / "experiments").is_dir())
        self.assertTrue((ROOT / "tools").is_dir())
        self.assertTrue((ROOT / "archive").is_dir())
        self.assertTrue((ROOT / "docs" / "foundations").is_dir())
        self.assertTrue((ROOT / "docs" / "research").is_dir())
        self.assertTrue((ROOT / "docs" / "results").is_dir())
        self.assertFalse((ROOT / "scripts").exists())
        self.assertFalse((ROOT / "adaptive_controller_system4").exists())
        self.assertFalse((ROOT / "baseline_hysteresis_v1").exists())
        self.assertFalse((ROOT / "intervention_engine_v1.5_v2").exists())
        self.assertFalse((ROOT / "v1.5").exists())
