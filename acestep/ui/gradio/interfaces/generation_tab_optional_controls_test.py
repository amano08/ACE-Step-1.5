"""Contract tests for generation optional-parameter controls."""

import ast
from pathlib import Path
import unittest


_OPTIONAL_PATH = Path(__file__).resolve().parent / "generation_tab_optional_controls.py"
_EXPECTED_NAMES = {
    "bpm",
    "key_scale",
    "time_signature",
    "vocal_language",
    "audio_duration",
}


_PRESET_VARS = {
    "bpm": "bpm_value",
    "key_scale": "keyscale_value",
    "time_signature": "timesig_value",
    "vocal_language": None,  # never preset; always starts locked
    "audio_duration": "duration_value",
}


class GenerationTabOptionalControlsTests(unittest.TestCase):
    """Verify optional controls start locked when Auto toggles are enabled."""

    def test_optional_fields_default_to_non_interactive(self):
        """Optional controls stay locked unless a launch preset pinned them.

        Fields the user pinned through ``ACESTEP_DEFAULT_*`` start editable with
        Auto unchecked, so their ``interactive=`` is an expression over that
        preset variable rather than a bare ``False``; the fields with no preset
        source must still be hardcoded non-interactive.
        """

        module = ast.parse(_OPTIONAL_PATH.read_text(encoding="utf-8"))
        func = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_optional_parameter_controls"
        )

        found: dict[str, ast.AST] = {}
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
            if target_name not in _EXPECTED_NAMES:
                continue
            if not isinstance(node.value, ast.Call):
                continue
            interactive_kw = next((kw for kw in node.value.keywords if kw.arg == "interactive"), None)
            if interactive_kw is not None:
                found[target_name] = interactive_kw.value

        self.assertEqual(_EXPECTED_NAMES, set(found.keys()))
        for field_name, expr in found.items():
            preset_var = _PRESET_VARS[field_name]
            if preset_var is None:
                self.assertIsInstance(expr, ast.Constant, f"{field_name} should use constant False")
                self.assertFalse(expr.value, f"{field_name} should default to non-interactive")
                continue
            names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
            self.assertIn(
                preset_var,
                names,
                f"{field_name} should derive interactive from {preset_var}",
            )


if __name__ == "__main__":
    unittest.main()
