"""Extract page-linked text and visual metadata from the OpenStax source PDF."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pymupdf


SOURCE_ID = "openstax-introductory-statistics-2e"
SOURCE_NAME = "Introductory Statistics 2e (OpenStax)"
SOURCE_URL = (
    "https://assets.openstax.org/oscms-prodcms/media/documents/"
    "introductory-statistics-2e_-_WEB.pdf"
)
FIGURE_RE = re.compile(r"\bFigure\s+(\d+(?:\.\d+)?[A-Za-z]?)\b", re.IGNORECASE)
TABLE_RE = re.compile(r"\bTable\s+(\d+(?:\.\d+)?[A-Za-z]?)\b", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_text(text: str, limit: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + limit // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _text_blocks(page: pymupdf.Page) -> list[dict[str, Any]]:
    blocks = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        text = "\n".join(
            "".join(span["text"] for span in line["spans"])
            for line in block.get("lines", [])
        ).strip()
        if text:
            blocks.append({"text": text, "bbox": [round(value, 2) for value in block["bbox"]]})
    return blocks


def _figure_captions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    captions = []
    for block in blocks:
        lines = [line.strip() for line in block["text"].splitlines() if line.strip()]
        for index, line in enumerate(lines):
            match = FIGURE_RE.search(line)
            if match:
                caption = " ".join(lines[index:index + 3])
                captions.append({
                    "figure_label": match.group(1),
                    "caption": caption[:700],
                    "bbox": block["bbox"],
                })
    return captions


def build_index(
    pdf_path: Path,
    output_dir: Path,
    *,
    chunk_chars: int,
    overlap: int,
    annotations_path: Path | None = None,
) -> dict[str, Any]:
    if chunk_chars < 200 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk size must be >=200 and overlap must be smaller than chunk size")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")
    annotations = []
    if annotations_path is not None:
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotation_by_key = {
        (item["pdf_page"], item["figure_label"]): item["description"]
        for item in annotations
    }

    document = pymupdf.open(pdf_path)
    chunks: list[dict[str, Any]] = []
    page_inventory = []
    visual_index = []
    total_chars = 0

    for page_index, page in enumerate(document):
        pdf_page = page_index + 1
        page_text = page.get_text("text", sort=True).strip()
        total_chars += len(page_text)
        blocks = _text_blocks(page)
        captions = _figure_captions(blocks)
        images = page.get_image_info(xrefs=True)
        drawings = page.get_drawings()
        table_records = []

        for chunk_index, text in enumerate(_split_text(page_text, chunk_chars, overlap)):
            chunks.append({
                "id": f"st2e-p{pdf_page:03d}-t{chunk_index:02d}",
                "text": text,
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "metadata": {
                    "pdf_page": pdf_page,
                    "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                    "content_type": "text",
                    "image_count": len(images),
                    "drawing_count": len(drawings),
                },
            })

        for table_index, table in enumerate(page.find_tables().tables):
            if table.row_count < 2 or table.col_count < 2:
                continue
            rows = [
                [str(cell).strip() if cell is not None else "" for cell in row]
                for row in table.extract()
            ]
            table_text = "\n".join(" | ".join(row) for row in rows)
            table_top = table.bbox[1]
            title_block = min(
                (block for block in blocks if TABLE_RE.search(block["text"])),
                key=lambda item: abs(item["bbox"][3] - table_top),
                default=None,
            )
            table_title = title_block["text"][:300] if title_block else ""
            table_record = {
                "id": f"st2e-p{pdf_page:03d}-tb{table_index:02d}",
                "pdf_page": pdf_page,
                "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                "bbox": [round(value, 2) for value in table.bbox],
                "title": table_title,
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "rows": rows,
            }
            table_records.append(table_record)
            chunks.append({
                "id": table_record["id"],
                "text": (
                    f"Structured table on PDF page {pdf_page}. "
                    f"{table_title}\n{table_text}"
                ),
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "metadata": {
                    "pdf_page": pdf_page,
                    "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                    "content_type": "table",
                    "bbox": table_record["bbox"],
                    "row_count": table_record["row_count"],
                    "column_count": table_record["column_count"],
                },
            })

        page_visuals = []
        for caption_index, caption in enumerate(captions):
            bbox = caption["bbox"]
            description = annotation_by_key.get(
                (pdf_page, caption["figure_label"]), ""
            )
            visual = {
                "id": f"st2e-p{pdf_page:03d}-v{caption_index:02d}",
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                "figure_label": caption["figure_label"],
                "caption": caption["caption"],
                "description": description,
                "bbox": bbox,
                "crop_bbox": [
                    0.0,
                    max(0.0, bbox[1] - 8.0),
                    round(page.rect.width, 2),
                    round(page.rect.height, 2),
                ],
                "image_count": len(images),
                "drawing_count": len(drawings),
                "source_url": SOURCE_URL,
            }
            page_visuals.append(visual)
            visual_index.append(visual)
            chunks.append({
                "id": visual["id"],
                "text": (
                    f"Figure {visual['figure_label']}. {visual['caption']}"
                    + (f"\nVisual description: {description}" if description else "")
                ),
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "metadata": {
                    "pdf_page": pdf_page,
                    "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                    "content_type": "visual_caption",
                    "figure_label": visual["figure_label"],
                    "bbox": bbox,
                    "crop_bbox": visual["crop_bbox"],
                    "image_count": len(images),
                    "drawing_count": len(drawings),
                },
            })

        image_assets = []
        for image_index, image in enumerate(images):
            image_bbox = [round(value, 2) for value in image["bbox"]]
            image_center_y = (image_bbox[1] + image_bbox[3]) / 2
            nearest_caption = min(
                captions,
                key=lambda item: abs(
                    image_center_y - (item["bbox"][1] + item["bbox"][3]) / 2
                ),
                default=None,
            )
            nearest_block = min(
                blocks,
                key=lambda item: abs(
                    image_center_y - (item["bbox"][1] + item["bbox"][3]) / 2
                ),
                default=None,
            )
            caption_text = nearest_caption["caption"] if nearest_caption else ""
            description = (
                annotation_by_key.get(
                    (pdf_page, nearest_caption["figure_label"]), ""
                )
                if nearest_caption
                else ""
            )
            nearby_text = (
                nearest_block["text"][:500]
                if nearest_block and nearest_block["text"] != caption_text
                else ""
            )
            asset = {
                "id": f"st2e-p{pdf_page:03d}-im{image_index:02d}",
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                "bbox": image_bbox,
                "xref": image.get("xref"),
                "width": image.get("width"),
                "height": image.get("height"),
                "figure_label": nearest_caption["figure_label"] if nearest_caption else None,
                "caption": caption_text,
                "description": description,
                "nearby_text": nearby_text,
                "source_url": SOURCE_URL,
            }
            image_assets.append(asset)
            visual_index.append(asset)
            visual_text = (
                f"Visual asset on PDF page {pdf_page}. "
                f"{caption_text or nearby_text or 'See the original page image.'}"
            )
            if nearby_text:
                visual_text += f"\nNearby text: {nearby_text}"
            if description:
                visual_text += f"\nVisual description: {description}"
            chunks.append({
                "id": asset["id"],
                "text": visual_text,
                "source_id": SOURCE_ID,
                "source": SOURCE_NAME,
                "page": pdf_page,
                "metadata": {
                    "pdf_page": pdf_page,
                    "book_page": pdf_page - 12 if pdf_page >= 13 else None,
                    "content_type": "image_asset",
                    "bbox": image_bbox,
                    "xref": asset["xref"],
                    "figure_label": asset["figure_label"],
                },
            })

        page_inventory.append({
            "pdf_page": pdf_page,
            "book_page": pdf_page - 12 if pdf_page >= 13 else None,
            "text_chars": len(page_text),
            "text_block_count": len(blocks),
            "image_count": len(images),
            "drawing_count": len(drawings),
            "tables": table_records,
            "image_assets": image_assets,
            "figure_captions": [item["figure_label"] for item in captions],
            "visuals": page_visuals,
        })

    stats = {
        "source_id": SOURCE_ID,
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "pdf_sha256": _sha256(pdf_path),
        "pdf_size_bytes": pdf_path.stat().st_size,
        "pdf_pages": len(document),
        "pages_with_extractable_text": sum(item["text_chars"] > 0 for item in page_inventory),
        "pages_without_extractable_text": [
            item["pdf_page"] for item in page_inventory if item["text_chars"] == 0
        ],
        "text_characters": total_chars,
        "image_instances": sum(item["image_count"] for item in page_inventory),
        "pages_with_images": sum(item["image_count"] > 0 for item in page_inventory),
        "drawing_instances": sum(item["drawing_count"] for item in page_inventory),
        "structured_tables": sum(len(item["tables"]) for item in page_inventory),
        "indexed_table_rows": sum(
            table["row_count"]
            for item in page_inventory
            for table in item["tables"]
        ),
        "figure_captions": sum(len(item["figure_captions"]) for item in page_inventory),
        "image_asset_records": sum(len(item["image_assets"]) for item in page_inventory),
        "indexed_chunks": len(chunks),
        "chunk_chars": chunk_chars,
        "chunk_overlap_chars": overlap,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "chunks.json": chunks,
        "page_inventory.json": page_inventory,
        "visual_index.json": visual_index,
        "source_stats.json": stats,
    }
    for name, value in files.items():
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    document.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--overlap-chars", type=int, default=160)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    stats = build_index(
        args.pdf,
        args.output,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap_chars,
        annotations_path=args.annotations,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
