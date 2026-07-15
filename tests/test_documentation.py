from datetime import date
import re
import unittest
from pathlib import Path

import maia2
import yaml


class DocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (
            Path(__file__).parents[1].joinpath("README.md").read_text(encoding="utf-8")
        )

    def test_readme_python_examples_are_syntactically_valid(self):
        blocks = re.findall(r"```python\n(.*?)```", self.readme, flags=re.DOTALL)
        self.assertTrue(blocks)
        for index, block in enumerate(blocks):
            with self.subTest(block=index):
                compile(block, f"README.md:python-block-{index}", "exec")

    def test_spawn_training_example_has_main_guard(self):
        training_section = self.readme.split("## Training", 1)[1].split(
            "## Interpretability", 1
        )[0]
        self.assertIn('if __name__ == "__main__":', training_section)

    def test_training_docs_name_both_strict_speed_presets(self):
        for name in (
            "maia2-training-rapid.yaml",
            "maia2-training-blitz.yaml",
        ):
            with self.subTest(name=name):
                self.assertIn(name, self.readme)
        self.assertIn("both `Rated` and `Rapid`", self.readme)
        self.assertIn("both `Rated` and `Blitz`", self.readme)

    def test_citation_version_matches_package_version(self):
        citation = yaml.safe_load(
            Path(__file__)
            .parents[1]
            .joinpath("CITATION.cff")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(citation["version"], maia2.__version__)
        date.fromisoformat(str(citation["date-released"]))

    def test_public_version_surfaces_match_package_version(self):
        root = Path(__file__).parents[1]
        version = maia2.__version__

        self.assertIn(
            f"https://img.shields.io/pypi/v/maia2.svg?v={version}",
            self.readme,
        )
        self.assertIn(
            f'placeholder: "{version} or commit SHA"',
            root.joinpath(".github", "ISSUE_TEMPLATE", "bug_report.yml").read_text(
                encoding="utf-8"
            ),
        )
        self.assertIn(
            f"assert maia2.__version__ == '{version}'",
            root.joinpath(".github", "workflows", "ci.yml").read_text(encoding="utf-8"),
        )

        release_series = ".".join(version.split(".")[:2])
        self.assertIn(
            f"| {release_series}.x | Yes |",
            root.joinpath("SECURITY.md").read_text(encoding="utf-8"),
        )

    def test_dependabot_keeps_runtime_version_updates_manual(self):
        config = yaml.safe_load(
            Path(__file__)
            .parents[1]
            .joinpath(".github", "dependabot.yml")
            .read_text(encoding="utf-8")
        )
        updates = config["updates"]
        pip_update = next(
            update for update in updates if update["package-ecosystem"] == "pip"
        )
        actions_update = next(
            update
            for update in updates
            if update["package-ecosystem"] == "github-actions"
        )

        self.assertEqual(pip_update["open-pull-requests-limit"], 0)
        self.assertNotIn("groups", pip_update)
        self.assertEqual(actions_update["schedule"]["interval"], "weekly")


if __name__ == "__main__":
    unittest.main()
