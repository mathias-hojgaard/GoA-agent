"""CLI entry point for the tender extraction pipeline."""

import asyncio
import json
import logging
import sys
from pathlib import Path

from src.models.ga_models import FileCoordinates
from src.pipeline import process_tender


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/run_tender.py <email.txt> <file_coordinates.json> [output.json]",
            file=sys.stderr,
        )
        sys.exit(1)

    email_path = sys.argv[1]
    fc_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.json"

    with open(email_path) as f:
        email_body = f.read()
    with open(fc_path) as f:
        file_coords = FileCoordinates(**json.load(f))

    result, updated_fc = await process_tender(email_body, file_coords)

    with open(output_path, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)

    out = Path(output_path)
    coords_path = str(out.with_name(out.stem + "_coordinates" + out.suffix))
    with open(coords_path, "w") as f:
        json.dump(updated_fc.model_dump(mode="json"), f, indent=2)

    print(f"Extracted {len(result.products)} product lines")
    print(f"Attributes: {list(result.attributes.extractions.keys())}")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
