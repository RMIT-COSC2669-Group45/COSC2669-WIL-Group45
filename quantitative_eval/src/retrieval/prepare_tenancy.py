from pathlib import Path
import re
import json

SOURCE_DIR = Path("data/tenancy")
OUTPUT = Path("data/tenancy_collection/collection.jsonl")


def clean_markdown(text):
    # Remove Markdown links but keep their visible text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove any remaining raw URLs.
    text = re.sub(r"https?://\S+", "", text)

    # Clean excessive whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def prepare_collection():
    files = sorted(SOURCE_DIR.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown files found in {SOURCE_DIR}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with OUTPUT.open("w", encoding="utf-8") as f:
        for source in files:
            text = source.read_text(encoding="utf-8")

            document_match = re.search(
                r"^document_id:\s*(.+)$",
                text,
                flags=re.MULTILINE
            )

            title_match = re.search(
                r'^title:\s*"?(.+?)"?$',
                text,
                flags=re.MULTILINE
            )

            document_id = (
                document_match.group(1).strip().strip('"')
                if document_match
                else source.stem.upper()
            )

            title = (
                title_match.group(1).strip().strip('"')
                if title_match
                else source.stem
            )

            # Remove YAML front matter.
            text = re.sub(
                r"^---\n.*?\n---\n",
                "",
                text,
                flags=re.DOTALL
            )

            # Split into heading-based chunks.
            parts = re.split(
                r"(?=^#{1,3}\s+)",
                text,
                flags=re.MULTILINE
            )

            document_chunks = 0

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                lines = part.splitlines()

                heading = lines[0].strip()
                body = "\n".join(lines[1:]).strip()

                # Ignore heading-only sections.
                if not body:
                    continue

                body = clean_markdown(body)

                document_chunks += 1
                total_chunks += 1

                contents = (
                    f"{title}\n\n"
                    f"{heading}\n\n"
                    f"{body}"
                )

                record = {
                    "id": f"{document_id}_{document_chunks:03d}",
                    "contents": contents
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

            print(
                f"{document_id}: "
                f"{document_chunks} chunks"
            )

    print()
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    prepare_collection()
