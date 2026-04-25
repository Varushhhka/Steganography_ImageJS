import subprocess
import sys
from pathlib import Path

# Чтобы работало
# Сначала установить NodeJS
# https://nodejs.org/en?spm=a2ty_o01.29997173.0.0.635155fbkEedyP
# Потом закинуть в проект файл package.json
# В терминальчике вввести: npm install


INPUT_DIR = "images_cats_512"
PAYLOAD_FILE = "stego_payload.txt"
OUTPUT_DIR = "stego_cats_512"
WORKER_SCRIPT = "worker.js"

LENGTHS = [50, 100, 500, 1_000]
IMAGES_COUNT = 1000


def ensure_dirs():
    Path(INPUT_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for length in LENGTHS:
        (Path(OUTPUT_DIR) / f"len_{length}").mkdir(exist_ok=True)


def load_hex_payloads(filepath, expected_lengths):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Файл {filepath} не найден!")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    hex_strings = [s.strip() for s in text.split() if s.strip()]
    if len(hex_strings) != len(expected_lengths):
        raise ValueError(f"В файле найдено {len(hex_strings)} последовательностей,а ожидалось {len(expected_lengths)}.")

    payloads = {}
    for length, hex_str in zip(expected_lengths, hex_strings):
        expected_hex_len = length * 2
        if len(hex_str) != expected_hex_len:
            raise ValueError(f"HEX для {length} байт имеет длину {len(hex_str)} символов,а ожидалось {expected_hex_len}.")
        try:
            bytes.fromhex(hex_str)
        except ValueError:
            raise ValueError(f"Строка для длины {length} содержит некорректные HEX-символы.")

        payloads[length] = hex_str
    return payloads


def run_js_worker(mode, args, payload_hex=None):
    cmd = ["node", str(Path(WORKER_SCRIPT).resolve()), mode] + [str(a) for a in args]
    input_data = payload_hex.encode() if payload_hex else None
    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=False,
        check=True,
        timeout=60,
        shell=False
    )
    return result.stdout.strip()


if __name__ == "__main__":
    ensure_dirs()

    try:
        payloads = load_hex_payloads(PAYLOAD_FILE, LENGTHS)
    except Exception as e:
        print(f"Ошибка загрузки payloads: {e}")
        sys.exit(1)

    all_images = list(Path(INPUT_DIR).glob("*.png"))
    if not all_images:
        print(f"В папке {INPUT_DIR} нет PNG изображений!")
        sys.exit(1)

    target_images = all_images[:]
    for length in LENGTHS:
        folder_name = f"len_{length}"
        out_dir = Path(OUTPUT_DIR) / folder_name
        payload_hex = payloads[length]

        print(f"\nДлина данных = {length} байт ({length * 8} бит)")
        success_count = 0
        test_file = None
        for img_path in target_images:
            out_path = out_dir / img_path.name
            run_js_worker("embed", [str(img_path), str(out_path)], payload_hex=payload_hex)
            success_count += 1
            if test_file is None:
                test_file = out_path
        print(f"Получилось: {success_count}/{len(target_images)} изображений в {folder_name}")

        if test_file and success_count > 0:
            print(f"Проверка извлечения: {test_file.name}")
            extracted_bytes_raw = run_js_worker("extract", [str(test_file), str(length * 8)])
            extracted_hex = extracted_bytes_raw.decode('utf-8')
            original_bytes = bytes.fromhex(payload_hex)
            extracted_bytes = bytes.fromhex(extracted_hex)
            if original_bytes == extracted_bytes:
                print(f"Данные извлечены корректно!")
            else:
                print(f"Данные не совпадают!")
            print(f"Оригинал: {original_bytes.hex()}")
            print(f"Извлечено: {extracted_bytes.hex()}")
