from pathlib import Path

INPUT_DIR = "image_cats_300x200"
PREFIX = "cat_image"
EXTENSION = ".png"


input_path = Path(INPUT_DIR)
files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == EXTENSION])
print(f"Найдено файлов: {len(files)}")

for index, old_path in enumerate(files, start=1):
    new_name = f"{PREFIX}{index:04d}{EXTENSION}"
    new_path = input_path / new_name
    if old_path.name == new_name:
        continue
    old_path.rename(new_path)

print(f"Переименовано.")