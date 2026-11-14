"""Download and verify the exact OpenStax PDF used by the advanced benchmark."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


SOURCE_METADATA = Path(__file__).with_name("source.json")


def download(output: Path, *, overwrite: bool = False) -> dict[str, str | int]:
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(metadata["source_pdf"], timeout=120) as response:
        with output.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
                digest.update(block)
                size += len(block)
    actual_hash = digest.hexdigest()
    expected_hash = metadata["pdf_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "Downloaded PDF checksum differs from the submitted benchmark source: "
            f"expected {expected_hash}, got {actual_hash}. Preserve this file and "
            "verify the publisher's version before updating source.json."
        )
    return {"path": str(output), "sha256": actual_hash, "size_bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.output, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
