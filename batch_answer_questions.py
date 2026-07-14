from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from main import (
    build_agent_and_retriever,
    extract_final_text,
    safe_agent_invoke,
    verify_final_answer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a sheet of questions using the LakeAgent retrieval pipeline.",
    )
    parser.add_argument(
        "--input-csv",
        default="[iSE Summer Challenge 2026] Questions - Sheet1.csv",
        help="CSV file containing question rows with columns STT and Question.",
    )
    parser.add_argument(
        "--output-csv",
        default="iSE_Summer_Challenge_2026_answers.csv",
        help="Output CSV path with columns id,answer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Deprecated compatibility flag. Answers are now written to the output CSV immediately.",
    )
    return parser


def answer_query(agent, verifier_model, query: str) -> str:
    response = safe_agent_invoke(agent, query)
    draft_answer = extract_final_text(response)
    return verify_final_answer(query, draft_answer, verifier_model)


def read_questions(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            question_id = (row.get("STT") or "").strip()
            question = (row.get("Question") or "").strip()
            if not question_id or not question:
                continue
            rows.append({"id": question_id, "question": question})
        return rows


def write_submission(output_path: Path, rows: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def read_existing_submission(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        existing: dict[str, str] = {}
        for row in reader:
            question_id = (row.get("id") or "").strip()
            answer = (row.get("answer") or "").strip()
            if question_id:
                existing[question_id] = answer
        return existing


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    questions = read_questions(input_path)
    if not questions:
        raise SystemExit("No valid question rows found in the input CSV.")

    existing_answers = read_existing_submission(output_path)
    submission_rows = []
    for question in questions:
        submission_rows.append(
            {
                "id": question["id"],
                "answer": existing_answers.get(question["id"], ""),
            }
        )
    row_index_by_id = {
        row["id"]: index
        for index, row in enumerate(submission_rows)
    }
    write_submission(output_path, submission_rows)
    print(f"Initialized submission template at {output_path}")

    agent, retriever, _model, verifier_model = build_agent_and_retriever()
    processed_count = 0

    try:
        for question in questions:
            existing_answer = submission_rows[row_index_by_id[question["id"]]]["answer"].strip()
            if existing_answer:
                processed_count += 1
                print(
                    f"[{processed_count}/{len(questions)}] skipped id={question['id']} (already answered)"
                )
                continue
            answer = answer_query(agent, verifier_model, question["question"])
            submission_rows[row_index_by_id[question["id"]]]["answer"] = answer
            write_submission(output_path, submission_rows)
            processed_count += 1
            print(f"[{processed_count}/{len(questions)}] answered id={question['id']} and wrote to {output_path}")
    finally:
        retriever.close()

    print(f"Completed: {processed_count}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
