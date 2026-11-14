"""Export Day 6 Chroma chunks and prepare an independent Day 7 benchmark."""
import argparse
import json
from pathlib import Path
from typing import Any

from baseline import hit
from convert_day6 import convert_question


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "principles-management"

NEW_TEST_QUESTIONS = [
    {
        "id": "q22",
        "question": "What six steps does the chapter list for the decision-making process?",
        "expected_answer": (
            "Recognize the decision, generate alternatives, analyze them, select one, "
            "implement it, and evaluate its effectiveness."
        ),
        "gold": [{
            "source_id": SOURCE_ID,
            "page": 42,
            "evidence": (
                "1. Recognize that a decision needs to be made.\n"
                "2. Generate multiple alternatives.\n"
                "3. Analyze the alternatives.\n"
                "4. Select an alternative.\n"
                "5. Implement the selected alternative.\n"
                "6. Evaluate its effectiveness."
            ),
        }],
    },
    {
        "id": "q23",
        "question": "What decision-making benefits does the chapter attribute to talking to other people?",
        "expected_answer": "It can improve quality by generating more options and reducing bias.",
        "gold": [{
            "source_id": SOURCE_ID,
            "page": 47,
            "evidence": "Talk to other people. Improves quality: generates more options, reduces bias",
        }],
    },
    {
        "id": "q24",
        "question": "How does the chapter define critical thinking?",
        "expected_answer": (
            "It is a disciplined process for evaluating information quality, especially "
            "data and arguments from other sources, to assess trustworthiness and validity."
        ),
        "gold": [{
            "source_id": SOURCE_ID,
            "page": 50,
            "evidence": (
                "Critical thinking is a disciplined process of evaluating the quality of "
                "information, especially data collected from other sources and arguments "
                "made by other people, to determine whether the source should be trusted "
                "or whether the argument is valid."
            ),
        }],
    },
    {
        "id": "q25",
        "question": "Which two major sets of leader behaviors did the Ohio State University studies identify?",
        "expected_answer": "Consideration and initiating structure.",
        "gold": [{
            "source_id": SOURCE_ID,
            "page": 390,
            "evidence": (
                "Their results identified two major sets of leader behaviors: "
                "consideration and initiating structure."
            ),
        }],
    },
    {
        "id": "q26",
        "question": "According to path-goal theory, what does an effective leader provide to organizational members?",
        "expected_answer": "A path to a valued goal.",
        "gold": [{
            "source_id": SOURCE_ID,
            "page": 395,
            "evidence": (
                "an effective leader provides organizational members with a path to a valued goal"
            ),
        }],
    },
]

UNANSWERABLE_QUESTIONS = [
    {
        "id": "dev-no-evidence-01",
        "split": "dev",
        "question": "What was Away's exact net profit margin during its first year?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The Away case reports first-year sales but does not state its net profit margin."
        ),
    },
    {
        "id": "dev-no-evidence-02",
        "split": "dev",
        "question": "How many Microsoft board members voted to approve the culture change?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The selected decision-making pages discuss Microsoft's culture change but "
            "contain no board vote or vote count."
        ),
    },
    {
        "id": "test-no-evidence-01",
        "split": "test",
        "question": "What was the Springfield Sea Lions' exact win-loss record in John Arroyo's season?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The leadership case describes Arroyo's plans and staff response but gives "
            "no season win-loss record."
        ),
    },
    {
        "id": "test-no-evidence-02",
        "split": "test",
        "question": "What percentage increase in employee satisfaction did Arroyo's plan produce?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "The Sea Lions case describes the staff response to Arroyo's plan but "
            "reports no measured employee-satisfaction percentage."
        ),
    },
]


def build_questions(day6_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for raw in day6_questions:
        question = convert_question(raw)
        # The former Day 6 test labels have already been inspected; use them only on dev.
        if question["split"] == "test":
            question["split"] = "dev"
        questions.append(question)
    questions.extend(
        {"answerable": True, **question, "split": "test"}
        for question in NEW_TEST_QUESTIONS
    )
    questions.extend(UNANSWERABLE_QUESTIONS)
    return questions


def validate_questions(chunks: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("Day 6 index returned no chunks")
    chunk_ids = set()
    for chunk in chunks:
        for field in ("id", "text", "source_id", "source", "page", "metadata"):
            if field not in chunk or chunk[field] is None:
                raise ValueError(f"chunk missing required field: {field}")
        if chunk["id"] in chunk_ids:
            raise ValueError(f"duplicate chunk id: {chunk['id']}")
        chunk_ids.add(chunk["id"])

    question_ids = set()
    for question in questions:
        if question["id"] in question_ids:
            raise ValueError(f"duplicate question id: {question['id']}")
        question_ids.add(question["id"])
        if question.get("answerable", True) is False:
            continue
        for evidence in question["gold"]:
            if not any(
                hit(chunk, [evidence])
                for chunk in chunks
            ):
                raise ValueError(
                    f"{question['id']}: evidence not found in indexed chunks "
                    f"on page {evidence['page']}"
                )

    for split, expected in (("dev", (15, 2)), ("test", (5, 2))):
        selected = [question for question in questions if question["split"] == split]
        answerable = sum(question.get("answerable", True) for question in selected)
        unanswerable = sum(question.get("answerable", True) is False for question in selected)
        if answerable < expected[0] or unanswerable < expected[1]:
            raise ValueError(
                f"{split} needs at least {expected[0]} answerable and "
                f"{expected[1]} unanswerable questions; got {answerable}/{unanswerable}"
            )


def export_chunks(index_path: Path, strategy: str) -> list[dict[str, Any]]:
    import chromadb

    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_collection(name=f"chunks_{strategy}")
    records = collection.get(include=["documents", "metadatas"])
    chunks = []
    for chunk_id, text, metadata in zip(
        records["ids"], records["documents"], records["metadatas"], strict=True
    ):
        if text is None or metadata is None:
            raise ValueError(f"Chroma returned incomplete data for chunk {chunk_id}")
        chunks.append({
            "id": chunk_id,
            "text": text,
            "source_id": metadata.get("document_id"),
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "metadata": metadata,
        })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day6-root", type=Path, required=True)
    parser.add_argument("--index-name", default="chroma_v2_b1000")
    parser.add_argument("--strategy", choices=("A", "B"), default="B")
    parser.add_argument(
        "--chunks-output", type=Path, default=ROOT / "starter" / "data" / "day6_chunks.json"
    )
    parser.add_argument(
        "--questions-output", type=Path, default=ROOT / "data" / "day7_questions.json"
    )
    args = parser.parse_args()
    day6_root = args.day6_root.resolve()
    index_path = day6_root / "starter" / "data" / args.index_name
    benchmark_path = day6_root / "starter" / "data" / "benchmark.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Day 6 benchmark not found: {benchmark_path}")

    chunks = export_chunks(index_path, args.strategy)
    day6_questions = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = build_questions(day6_questions)
    validate_questions(chunks, questions)

    args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
    args.questions_output.parent.mkdir(parents=True, exist_ok=True)
    args.chunks_output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.questions_output.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Exported {len(chunks)} chunks and {len(questions)} questions; "
        f"questions saved to {args.questions_output}"
    )


if __name__ == "__main__":
    main()
