import unittest

from docsearch.index import build_doc, group_blocks


class ReadingOrderTest(unittest.TestCase):
    def test_chunks_preserve_effluent_values_vertical_order(self):
        specs = [
            (1749, 369, 564, "二、計畫水質："),
            (1785, 421, 662, "A. 設計原污水水質"),
            (1814, 477, 719, "BOD: 200 mg/!"),
            (1845, 476, 719, "COD: 400 mg/1"),
            (1875, 475, 718, "SS : 200 mg/l"),
            (1909, 475, 952, "大腸菌類：10000000 CFU/100ml"),
            (1949, 423, 639, "B. 設計放流水質"),
            (1979, 477, 714, "BOD< 30 mg/l"),
            (2013, 478, 731, "COD< 100 mg/l"),
            (2040, 474, 732, "SS < 30 mg/l"),
            (2075, 476, 929, "大腸菌類< 200000 CFU/100ml"),
            (500, 100, 300, "圖面標題"),
            (600, 100, 300, "其他設計說明"),
            (700, 100, 300, "備註資料"),
        ]
        lines = [
            {"text": text, "bbox": [x0, y, x1, y + 40]}
            for y, x0, x1, text in specs
        ]
        blocks = group_blocks(lines[:11])
        ordered = sorted(range(11), key=lambda i: (lines[i]["bbox"][1] + lines[i]["bbox"][3]) / 2)
        for block in blocks:
            members = sorted(block, key=lambda i: ordered.index(i))
            for left, right in zip(members, members[1:]):
                a, b = lines[left]["bbox"], lines[right]["bbox"]
                for c_idx in set(ordered[ordered.index(left) + 1:ordered.index(right)]):
                    c = lines[c_idx]["bbox"]
                    cy = (c[1] + c[3]) / 2
                    self.assertFalse(
                        (a[1] + a[3]) / 2 < cy < (b[1] + b[3]) / 2
                        and min(a[2], c[2]) > max(a[0], c[0])
                        and min(b[2], c[2]) > max(b[0], c[0])
                    )

        chunks, _ = build_doc({"doc": "d", "lines": lines, "size": [1200, 3000]})
        expected = [
            "A. 設計原污水水質", "BOD: 200", "COD: 400", "SS : 200",
            "大腸菌類：", "B. 設計放流水質", "BOD< 30", "COD< 100", "SS < 30",
        ]
        matches = [
            chunk["body"] for chunk in chunks
            if chunk["kind"] in {"section", "neighbourhood"}
            and expected[0] in chunk["body"] and expected[5] in chunk["body"]
        ]
        self.assertTrue(matches, "expected a section or neighbourhood containing both headings")
        for body in matches:
            present = [body.index(text) for text in expected if text in body]
            self.assertEqual(present, sorted(present), body)
        self.assertTrue(any(
            "B. 設計放流水質" in chunk["body"]
            and "BOD< 30" in chunk["body"]
            and chunk["body"].index("BOD< 30") > chunk["body"].index("B. 設計放流水質")
            for chunk in chunks
        ))


if __name__ == "__main__":
    unittest.main()
