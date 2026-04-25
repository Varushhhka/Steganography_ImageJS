import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
from PIL import Image
import math
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


class LSBStegoAnalyzer:
    def __init__(self, original_path, stego_path):
        self.original_path = original_path
        self.stego_path = stego_path
        self.orig_rgb = np.array(Image.open(original_path).convert('RGB'))
        self.stego_rgb = np.array(Image.open(stego_path).convert('RGB'))
        self.orig_gray = np.array(Image.open(original_path).convert('L'))
        self.stego_gray = np.array(Image.open(stego_path).convert('L'))

    def calculate_quality(self):
        mse = np.mean((self.orig_rgb - self.stego_rgb) ** 2)
        psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
        ssim_val = ssim(self.orig_rgb, self.stego_rgb, channel_axis=2, data_range=255)

        metrics = {'PSNR': psnr, 'MSE': mse, 'SSIM': ssim_val}
        return metrics

    def interpret_metrics(self, m):
        interp = {}
        interp['PSNR'] = "Отличное (>40)" if m['PSNR'] > 40 else "Хорошее (35-40)" if m['PSNR'] > 35 else "Удовл. (30-35)" if m['PSNR'] > 30 else "Плохо (<30)"
        interp['MSE'] = "Отлично (<1.0)" if m['MSE'] < 1.0 else "Очень хорошо (1-3)" if m['MSE'] < 3.0 else "Хорошо (3-6)" if m['MSE'] < 6.0 else "Плохо (>10)"
        interp['SSIM'] = "Отлично (>0.98)" if m['SSIM'] > 0.98 else "Очень хорошо (0.95-0.98)" if m['SSIM'] > 0.95 else "Хорошо (0.90-0.95)" if m['SSIM'] > 0.90 else "Плохо"
        return interp

    def visualize_all(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # 1. Оригинал и Стего
        axes[0, 0].imshow(self.orig_rgb)
        axes[0, 0].set_title("Оригинальное изображение")

        axes[0, 1].imshow(self.stego_rgb)
        axes[0, 1].set_title("Стего-контейнер (ImageJS)")

        # 2. Разностное изображение (усиленное для видимости)
        # Показывает, где конкретно LSB изменил пиксели
        diff = np.abs(self.orig_gray.astype(np.int16) - self.stego_gray.astype(np.int16))
        axes[0, 2].imshow(diff * 255, cmap='hot')  # Умножаем на 255, чтобы увидеть разницу в 1 бит
        axes[0, 2].set_title("Карта изменений (Residual)")

        # 3. Анализ битовых плоскостей (LSB Plane)
        lsb_plane = np.bitwise_and(self.stego_gray, 1)
        axes[1, 0].imshow(lsb_plane, cmap='gray')
        axes[1, 0].set_title("Младший битовый слой (LSB)")

        # 4. Гистограммы
        axes[1, 1].hist(self.orig_gray.flatten(), bins=256, alpha=0.5, label='Оригинал', color='blue')
        axes[1, 1].hist(self.stego_gray.flatten(), bins=256, alpha=0.5, label='Стего', color='red')
        axes[1, 1].set_title("Сравнение гистограмм")
        axes[1, 1].legend()

        # 5. Разность гистограмм
        hist_orig, _ = np.histogram(self.orig_gray, bins=256, range=(0, 256))
        hist_stego, _ = np.histogram(self.stego_gray, bins=256, range=(0, 256))
        hist_diff = hist_stego - hist_orig

        axes[1, 2].bar(range(256), hist_diff, color='purple')
        axes[1, 2].set_title("Дельта частот пикселей")

        plt.show()

    def run_full_report(self):
        m = self.calculate_quality()
        i = self.interpret_metrics(m)

        print("-" * 30)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА ВНЕДРЕНИЯ:")
        for key in m:
            print(f"{key:5}: {m[key]:8.4f} | Оценка: {i[key]}")
        print("-" * 30)

        self.visualize_all()


def batch_analyze(orig_dir, stego_dir):
    orig_files = sorted(glob.glob(os.path.join(orig_dir, "*.png")))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, "*.png")))

    results = []
    pairs = list(zip(orig_files, stego_files))

    for orig_path, stego_path in tqdm(pairs, desc="Пакетный анализ"):
        try:
            analyzer = LSBStegoAnalyzer(orig_path, stego_path)
            metrics = analyzer.calculate_quality()
            results.append(metrics)
        except Exception as e:
            print(f"\nОшибка {os.path.basename(orig_path)}: {e}")
            continue

    df = pandas.DataFrame(results)
    print("\n=== СВОДНАЯ СТАТИСТИКА ===")
    print(df[['PSNR', 'MSE', 'SSIM']].mean())
    df.to_csv("stego_metrics_batch.csv", index=False)
    return df


analyzer = LSBStegoAnalyzer('images_cats_512/cat_image0001.png', 'stego_cats_512/len_1000/cat_image0001.png')
analyzer.run_full_report()
batch_analyze('images_cats_512', 'stego_cats_512/len_50')
batch_analyze('images_cats_512', 'stego_cats_512/len_100')
batch_analyze('images_cats_512', 'stego_cats_512/len_5000')
batch_analyze('images_cats_512', 'stego_cats_512/len_1000')
batch_analyze('images_cats_512', 'stego_cats_512/len_50000')