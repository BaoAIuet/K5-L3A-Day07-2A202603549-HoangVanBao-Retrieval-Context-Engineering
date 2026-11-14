import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "submission" / "advanced"))

from run_agent import ReferenceAwareTfidfIndex
from validate_benchmark import _validate_multi_hop


def _chunk(chunk_id, text, content_type, *, page, label=None):
    metadata = {"content_type": content_type}
    if label is not None:
        metadata["figure_label"] = label
    return {
        "id": chunk_id,
        "text": text,
        "source_id": "book",
        "source": "Statistics",
        "page": page,
        "metadata": metadata,
    }


class ReferenceAwareTfidfIndexTests(unittest.TestCase):
    def test_promotes_exactly_named_table(self):
        chunks = [
            _chunk(
                "wrong-table",
                "Table 4.21 | x | P(x) | 3 | 0.40",
                "table",
                page=276,
            ),
            _chunk(
                "target-table",
                "Table 4.20 | x | P(x) | 3 | 0.15",
                "table",
                page=276,
            ),
        ]

        results = ReferenceAwareTfidfIndex(chunks).search(
            "In Table 4.20, what is the probability assigned to x = 3?",
            top_k=1,
        )

        self.assertEqual([item["id"] for item in results], ["target-table"])

    def test_returns_chunks_for_each_named_figure(self):
        chunks = [
            _chunk("figure-216", "Figure 2.16 symmetric histogram", "visual_caption",
                   page=116, label="2.16"),
            _chunk("figure-218", "Figure 2.18 right-skewed histogram", "visual_caption",
                   page=117, label="2.18"),
            _chunk("decoy", "Histogram distributions and shape", "text", page=500),
        ]

        results = ReferenceAwareTfidfIndex(chunks).search(
            "Compare Figures 2.16 and 2.18",
            top_k=2,
        )

        self.assertEqual(
            {item["id"] for item in results},
            {"figure-216", "figure-218"},
        )

    def test_promotes_the_named_key_terms_section(self):
        chunks = [
            _chunk("general", "Relative frequency is discussed in the chapter.",
                   "text", page=40),
            _chunk("key-terms", "Key Terms\nRelative Frequency: a ratio of counts.",
                   "text", page=55),
        ]

        results = ReferenceAwareTfidfIndex(chunks).search(
            "Use the relative frequency definition in Key Terms",
            top_k=1,
        )

        self.assertEqual([item["id"] for item in results], ["key-terms"])

    def test_multi_hop_accepts_two_tables_on_one_page(self):
        question = {
            "id": "multi-table",
            "gold": [
                {"page": 603, "evidence_ref": "table:11.27"},
                {"page": 603, "evidence_ref": "table:11.28"},
            ],
        }

        _validate_multi_hop(question)

    def test_multi_hop_rejects_repeated_evidence_on_one_page(self):
        question = {
            "id": "not-multi-hop",
            "gold": [
                {"page": 603, "evidence": "grade A"},
                {"page": 603, "evidence": "grade C"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "at least 2 pages, tables, or figures"):
            _validate_multi_hop(question)


if __name__ == "__main__":
    unittest.main()
