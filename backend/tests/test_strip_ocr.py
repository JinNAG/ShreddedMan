import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strip_ocr import parse_tesseract_tsv


class TesseractParsingTests(unittest.TestCase):
    def test_quote_glyph_does_not_swallow_later_tsv_rows(self):
        header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
        def line(number, word):
            return f"5\t1\t1\t1\t1\t{number}\t10\t20\t30\t40\t90\t{word}"
        output = "\n".join((header, line(1, '"'), line(2, "mission"), line(3, "squad")))
        words = parse_tesseract_tsv(output)
        self.assertEqual([word["text"] for word in words], ['"', "mission", "squad"])
        self.assertEqual(words[2]["box"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
