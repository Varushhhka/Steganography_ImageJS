import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==================== НАСТРОЙКИ ПУТЕЙ И ПОРОГА ====================
MODEL_PATH = "aletheia/aletheia-models/effnetb0-A-alaska2-lsbr.h5"
ALETHEIA_BIN = "aletheia/aletheia.py"
THRESHOLD = 0.05  # Снизили порог для повышения чувствительности

DATASETS = {
    "clean": "512x512/images_cats",
    "len_50": "512x512/len_50",
    "len_100": "512x512/len_100",
    "len_500": "512x512/len_500",
    "len_1000": "512x512/len_1000",
    "len_50000": "al/stego_cats_long/stego_cats_long/len_50000"
}

OUTPUT_CSV = "stego_detailed_results.csv"
METRICS_CSV = "stego_metrics_summary.csv"
PLOT_IMAGE = "stego_efficiency_plot.png"


# ===================================================================

def run_effnet_predict(folder_path):
    """Выполняет команду effnetb0-predict и возвращает словарь {файл: вероятность}"""
    cmd = f"python {ALETHEIA_BIN} effnetb0-predict {folder_path} {MODEL_PATH} CPU"
    print(f"--> Анализ папки: {folder_path}")

    try:
        raw_output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"!!! Ошибка: {e}")
        return {}

    results = {}
    matches = re.findall(r"([^\s]+\.(?:png|jpg|jpeg|bmp))\s+([\d\.]+)", raw_output, re.IGNORECASE)
    for path, prob in matches:
        fname = os.path.basename(path)
        results[fname] = float(prob)
    return results


def main():
    all_data = []

    # 1. Сбор данных
    for label, folder_path in DATASETS.items():
        if not os.path.exists(folder_path):
            print(f"[-] Пропуск: Путь не найден: {folder_path}")
            continue

        predictions = run_effnet_predict(folder_path)

        for fname, prob in predictions.items():
            all_data.append({
                "Dataset": label,
                "Filename": fname,
                "Probability": prob,
                "True_Label": 0 if label == "clean" else 1,
                "Is_Detected": 1 if prob > THRESHOLD else 0
            })

    if not all_data:
        print("\n[!] Данные не собраны. Проверьте пути.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # 2. Расчет расширенных метрик
    summary_list = []
    for label in DATASETS.keys():
        subset = df[df['Dataset'] == label]
        if subset.empty: continue

        y_true = subset['True_Label']
        y_pred = subset['Is_Detected']

        # Матрица ошибок
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Расчет метрик
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Ложная тревога и Пропуск
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        summary_list.append({
            "Folder": label,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "FPR (Ложная тревога)": round(fpr, 4),
            "FNR (Пропуск цели)": round(fnr, 4),
            "Avg_Prob": round(subset['Probability'].mean(), 4),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

    metrics_df = pd.DataFrame(summary_list)
    metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8')

    print("\n--- СВОДНЫЕ МЕТРИКИ (Порог: {}) ---".format(THRESHOLD))
    print(metrics_df[['Folder', 'Accuracy', 'F1-Score', 'FPR (Ложная тревога)', 'FNR (Пропуск цели)', 'Avg_Prob']])

    # 3. Визуализация
    plt.figure(figsize=(12, 6))

    # Линия средней уверенности
    plt.plot(metrics_df['Folder'], metrics_df['Avg_Prob'], marker='o', linewidth=3, label='Средняя вероятность',
             color='#2c3e50')
    # Столбцы точности
    plt.bar(metrics_df['Folder'], metrics_df['Accuracy'], alpha=0.3, color='#3498db', label='Точность (Accuracy)')
    # Линия F1-Score
    plt.plot(metrics_df['Folder'], metrics_df['F1-Score'], marker='s', linestyle='--', color='#e67e22',
             label='F1-Score')

    plt.axhline(y=THRESHOLD, color='red', linestyle=':', label=f'Порог {THRESHOLD}')
    plt.title(f"Анализ эффективности стегодетектора (Порог {THRESHOLD})", fontsize=14)
    plt.ylabel("Значение метрик")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(PLOT_IMAGE, dpi=300)
    print(f"\n[OK] Результаты: {OUTPUT_CSV}, {METRICS_CSV}, {PLOT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    main()