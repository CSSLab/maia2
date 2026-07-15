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


if __name__ == "__main__":
    unittest.main()
