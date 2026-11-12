"""Offline checks for the two pieces that stop a 4B model from getting drawing arithmetic wrong.

Run: python -m unittest discover -s tests
"""
import unittest

from docsearch import calc
from docsearch.index import dim_notes


class Evaluate(unittest.TestCase):
    def test_volume_of_a_tank(self):
        self.assertAlmostEqual(calc.evaluate("7.9*3.525*3.2"), 89.112, places=3)

    def test_drawing_operators(self):
        self.assertAlmostEqual(calc.evaluate("4.675×7.9×1.85"), 68.325125, places=3)
        self.assertEqual(calc.evaluate("425 x 12 ÷ 24"), 212.5)

    def test_rejects_anything_but_arithmetic(self):
        for bad in ("__import__('os')", "open('x')", "2**99", "1/0", "a+1"):
            with self.assertRaises(ValueError, msg=bad):
                calc.evaluate(bad)


class Fill(unittest.TestCase):
    def test_placeholder_is_replaced_and_the_models_own_result_dropped(self):
        self.assertEqual(calc.fill("槽(1): {{7.9*3.525*3.2}} = 50.9 m³"), "槽(1): 7.9*3.525*3.2 = 89.11 m³")

    def test_unevaluable_placeholder_keeps_the_text(self):
        self.assertEqual(calc.fill("{{foo(1)}}"), "foo(1)")

    def test_wrong_product_written_by_the_model_is_corrected(self):
        out = calc.fill("7.9×3.52×1.85 = 50.4 m³")
        self.assertIn("= 51.44", out)
        self.assertIn("50.4", out)           # the correction names what the model wrote

    def test_right_product_is_left_alone(self):
        text = "7.9×3.525×3.2 = 89.11 m³ and 4.675×3×3.2 = 44.88 m³"
        self.assertEqual(calc.fill(text), text)

    def test_quoted_drawing_formulas_are_not_rewritten(self):
        # sum of terms and a quoted ratio: the drawing's own text, possibly cut off by OCR - not the model's arithmetic
        for text in ("7.9x3.52x1.85x2+4.675x7.9x1.85x2=239.5m3", "V=425m3/日×12÷24=212.5m3", "停留時間：223.12÷410×24= 13.1(hr)"):
            self.assertEqual(calc.fill(text), text, text)

    def test_dates_and_ids_are_not_arithmetic(self):
        self.assertEqual(calc.fill("112.01.05 id 3.5x6"), "112.01.05 id 3.5x6")


class TankSizes(unittest.TestCase):
    def test_three_tanks_of_the_calculation_sheet(self):
        body = ("設計尺寸：\n(1) L790×W352.5×H350/ D320\n(2) L790×W352.5×H350/ D320\n(3) L467.5×W300×H350/ D320")
        notes = dim_notes(body).splitlines()
        self.assertEqual(len(notes), 3)
        self.assertIn("(1)", notes[0]); self.assertIn("89.11 m3", notes[0])
        self.assertIn("(3)", notes[2]); self.assertIn("44.88 m3", notes[2])

    def test_text_without_sizes_gets_no_note(self):
        self.assertEqual(dim_notes("1. 初沉池\n設計水量：Q=425 CMD"), "")


if __name__ == "__main__":
    unittest.main()
