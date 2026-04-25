import os
import sys
import numpy as np
from PIL import Image
from scipy.stats import chi2 as chi2_dist
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def chi2_test_pov(histogram):
    even = histogram[0::2].astype(float)
    odd = histogram[1::2].astype(float)

    expected = (even + odd) / 2.0
    observed = even

    valid = expected > 0
    if not np.any(valid):
        return 0.0, 1.0

    observed = observed[valid]
    expected = expected[valid]

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    dof = max(len(observed) - 1, 1)

    p_value = 1.0 - chi2_dist.cdf(chi2_stat, dof)
    return chi2_stat, float(p_value)


def calculate_block_size(width, height, percent=5.0):
    total_pixels = width * height
    block_pixels = int(total_pixels * percent / 100.0)
    block_pixels = max(block_pixels, 32 * 32)
    block_size = int(np.sqrt(block_pixels))
    return min(block_size, min(width, height))


def analyze_image_chi2(img_path, block_percent=5.0):
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        h, w, _ = img_array.shape

        block_size = calculate_block_size(w, h, block_percent)

        block_p_values = []
        block_chi2_values = []

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                block = img_array[y:y_end, x:x_end, :]

                block_channel_ps = []
                block_channel_chi2 = []
                for ch in range(3):
                    channel_data = block[:, :, ch]
                    hist = np.bincount(channel_data.ravel(), minlength=256)
                    chi2_stat, p = chi2_test_pov(hist)
                    block_channel_ps.append(p)
                    block_channel_chi2.append(chi2_stat)

                avg_block_p = np.mean(block_channel_ps)
                avg_block_chi2 = np.mean(block_channel_chi2)
                block_p_values.append(avg_block_p)
                block_chi2_values.append(avg_block_chi2)

        if not block_p_values:
            return {'file': os.path.basename(img_path), 'avg_p': 0.0, 'avg_chi2': 0.0, 'error': 'Нет блоков'}

        final_avg_p = np.mean(block_p_values)
        final_avg_chi2 = np.mean(block_chi2_values)

        return {
            'file': os.path.basename(img_path),
            'avg_p': final_avg_p,
            'avg_chi2': final_avg_chi2,
            'block_size': block_size,
            'num_blocks': len(block_p_values),
            'error': None
        }

    except Exception as e:
        return {
            'file': os.path.basename(img_path),
            'avg_p': 0.0,
            'avg_chi2': 0.0,
            'block_size': 0,
            'num_blocks': 0,
            'error': str(e)
        }


def interpret_result(p):
    if p > 0.75:
        return "ВСТРАИВАНИЕ"
    elif p > 0.5:
        return "ПОДОЗРИТЕЛЬНО"
    else:
        return "ЧИСТОЕ"


def process_single_file(args):
    file_path, block_percent = args
    res = analyze_image_chi2(file_path, block_percent)
    res['status'] = interpret_result(res['avg_p']) if res['error'] is None else 'ОШИБКА'
    return res


def get_png_files(folder_path):
    png_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.png'):
                png_files.append(os.path.join(root, f))
    return sorted(png_files)


def main():
    block_percent = 5.0

    folder_path = input("Путь к папке с PNG: ").strip().strip('"\'')
    if not os.path.isdir(folder_path):
        print(f"Ошибка: папка '{folder_path}' не существует.")
        sys.exit(1)

    png_files = get_png_files(folder_path)
    total_files = len(png_files)
    if total_files == 0:
        print("PNG-файлы не найдены.")
        sys.exit(1)

    print(f"Найдено файлов: {total_files} | Размер блока: ~{block_percent}% площади")
    print("Запуск блочного анализа...\n")

    max_workers = min(6, os.cpu_count() or 4)
    results = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, (fp, block_percent)): fp for fp in png_files}
        for future in as_completed(future_to_file):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == total_files:
                if res['error'] is None:
                    print(f"[{completed:4d}/{total_files}] {res['file']:<30} p={res['avg_p']:.4f} χ²={res['avg_chi2']:.2f} ({res['status']})")
                else:
                    print(f"[{completed:4d}/{total_files}] {res['file']:<30} ({res['status']})")

    results.sort(key=lambda x: x['file'])
    valid = [r for r in results if r['error'] is None]

    print(f"Первые 100 файлов\n")
    print(f"{'№':<4} {'Файл':<30} {'Avg P-Value':<12} {'Avg χ²':<12} {'Статус':<15}")
    print("-" * 100)
    for i, r in enumerate(valid[:100]):
        print(f"{i + 1:<4} {r['file']:<30} {r['avg_p']:<12.4f} {r['avg_chi2']:<12.2f} {r['status']:<15}")

    categories = {"ЧИСТОЕ": 0, "ПОДОЗРИТЕЛЬНО": 0, "ВСТРАИВАНИЕ": 0}
    for r in valid:
        if r['status'] in categories:
            categories[r['status']] += 1
    print(f"\n{'-' * 100}")
    print("Общий результат:\n")
    print(f"Всего изображений обработано: {len(valid)}")
    print("\nРаспределение по категориям:")
    for cat, cnt in categories.items():
        pct = 100 * cnt / max(len(valid), 1)
        print(f"  {cat}: {cnt:4d} ({pct:5.1f}%)")

    if valid:
        ps = [r['avg_p'] for r in valid]
        chi2s = [r['avg_chi2'] for r in valid]
        print(f"  Среднее значение p-value: {np.mean(ps):.4f}")
        print(f"  Среднее значение χ²: {np.mean(chi2s):.2f}")

    print(f"\nВремя выполнения: {time.time() - start_time:.2f} сек")


if __name__ == "__main__":
    main()