import subprocess
import re
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== НАСТРОЙКИ ====================
MAX_WORKERS = 10
PYTHON_PATH = sys.executable
ALETHEIA_BIN = "aletheia/aletheia.py"
THRESHOLD = 0.05

DATASETS = {
    "clean": "512x512/images_cats",
    "len_50": "512x512/len_50",
    "len_100": "512x512/len_100",
    "len_500": "512x512/len_500",
    "len_1000": "512x512/len_1000",
    "len_50000": "al/stego_cats_long/stego_cats_long/len_50000"
}

OUTPUT_CSV = "ws_results.csv"


# ===================================================

def analyze_single_file(file_info):
    label, folder_path, filename = file_info
    file_path = os.path.join(folder_path, filename)

    # Флаг --fix убран по твоему запросу
    cmd = [PYTHON_PATH, ALETHEIA_BIN, "ws", file_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', shell=False)
        raw_output = result.stdout

        # Ищем значения каналов
        channel_values = re.findall(r"channel [RGB]\s+([\d\.]+)", raw_output)

        if channel_values:
            val = sum(float(v) for v in channel_values) / len(channel_values)
        else:
            # Если каналы не найдены (например, из-за ошибки 4-го канала), пишем 0
            val = 0.0

        return [label, filename, val, 0 if label == "clean" else 1, 1 if val > THRESHOLD else 0]
    except Exception:
        return [label, filename, 0.0, 0 if label == "clean" else 1, 0]


def main():
    all_tasks = []
    for label, folder_path in DATASETS.items():
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                all_tasks.append((label, folder_path, f))

    if not all_tasks:
        print("[!] Файлы не найдены.")
        return

    print(f"[*] Всего файлов: {len(all_tasks)}")
    print(f"[*] Пишем результаты в {OUTPUT_CSV}...")

    # Создаем/очищаем файл и пишем заголовок
    header = ["Dataset", "Filename", "Value", "True_Label", "Is_Detected"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    results_for_plot = []

    # Чтобы не открывать/закрывать файл 5000 раз, открываем один раз с буферизацией
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8", buffering=1) as csv_file:
        writer = csv.writer(csv_file)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(analyze_single_file, task): task for task in all_tasks}

            for i, future in enumerate(as_completed(future_to_file), 1):
                res = future.result()
                if res:
                    writer.writerow(res)
                    results_for_plot.append(res)

                if i % 10 == 0 or i == len(all_tasks):
                    print(f"\rОбработано: {i}/{len(all_tasks)} ({(i / len(all_tasks) * 100):.1f}%)", end="", flush=True)

    print(f"\n\n[+] Готово. Данные в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()