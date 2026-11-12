"""Job folders created by the video analysis (step 1) and used by view generation (step 2)."""

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .models import EgoConfig
from .store import Store, now, numbered_directory


def copy_image(source: Path, destination: Path):
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((4096, 4096))
        image.save(destination, format="PNG")


def contact_sheet(paths: list[Path], output: Path):
    sheet = Image.new("RGB", (320 * min(4, len(paths)), 264 * ((len(paths) + 3) // 4)), "white")
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert("RGB"), (320, 240))
        x, y = i % 4 * 320, i // 4 * 264
        sheet.paste(tile, (x + (320 - tile.width) // 2, y))
        ImageDraw.Draw(sheet).text((x + 5, y + 244), f"reference {i + 1}", fill="black")
    sheet.save(output)


def create_job(store: Store, references: list[Path], description: str, config: EgoConfig) -> dict:
    if not 1 <= len(references) <= 8:
        raise ValueError("Select 1 to 8 object reference images")
    if not 1 <= len(description.strip()) <= 4000:
        raise ValueError("Describe the target object in 1 to 4000 characters")
    directory = numbered_directory(store.root, "job")
    job_id = directory.name
    try:
        (directory / "references").mkdir()
        copied = []
        for i, source in enumerate(references):
            target = directory / "references" / f"{i + 1:02d}.png"
            copy_image(source, target)
            copied.append(store.artifact(job_id, target))
        contact_sheet([store.resolve(job_id, a) for a in copied], directory / "references.png")
        job = {
            "schema_version": 1,
            "id": job_id,
            "created_at": now(),
            "status": "ready",
            "description": description,
            "config": config.model_dump(),
            "references": copied,
            "reference_sheet": store.artifact(job_id, directory / "references.png"),
        }
        store.save(job)
        return job
    except BaseException:
        shutil.rmtree(directory)
        raise
