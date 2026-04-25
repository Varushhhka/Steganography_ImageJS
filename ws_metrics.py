import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# ==================== НАСТРОЙКИ ====================
INPUT_CSV = "ws_results.csv"
THRESHOLD = 0.05


# ===================================================

def build_combined_report():
    if not pd.io.common.file_exists(INPUT_CSV):
        print(f"[!] Файл {INPUT_CSV} не найден.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['Predicted_Label'] = (df['Value'] > THRESHOLD).astype(int)

    # Собираем метрики по каждому датасету отдельно
    stats = []
    datasets = df['Dataset'].unique()

    # Сортируем датасеты, чтобы clean был первым, а потом по возрастанию длины
    # (если названия позволяют простую сортировку)
    datasets = sorted(datasets, key=lambda x: (x != 'clean', x))

    for ds in datasets:
        sub = df[df['Dataset'] == ds]
        acc = accuracy_score(sub['True_Label'], sub['Predicted_Label'])
        f1 = f1_score(sub['True_Label'], sub['Predicted_Label'], zero_division=0)
        mean_val = sub['Value'].mean()

        stats.append({
            "Dataset": ds,
            "Accuracy": acc,
            "F1-Score": f1,
            "Mean_Value": mean_val
        })

    report_df = pd.DataFrame(stats).set_index("Dataset")

    # --- СТРОИМ ГРАФИК ---
    plt.figure(figsize=(12, 6))

    # 1. Столбцы (Accuracy)
    plt.bar(report_df.index, report_df['Accuracy'], color='#c6e2f5', label='Точность (Accuracy)', alpha=0.8)

    # 2. Линия F1-Score
    plt.plot(report_df.index, report_df['F1-Score'], color='#e67e22', marker='s',
             linestyle='--', label='F1-Score')

    # 3. Линия Средней вероятности (Mean Value)
    plt.plot(report_df.index, report_df['Mean_Value'], color='#2c3e50', marker='o',
             linewidth=3, label='Средняя вероятность')

    # 4. Порог
    plt.axhline(y=THRESHOLD, color='red', linestyle=':', label=f'Порог {THRESHOLD}')

    # Оформление
    plt.title(f"Анализ эффективности стегодетектора (Порог {THRESHOLD})", fontsize=14)
    plt.ylabel("Значение метрик")
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    plt.legend(loc='upper right')

    # Чтобы график выглядел аккуратно
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig("ws_final_report.png", dpi=300)
    print("[OK] График сохранен в ws_final_report.png")

    # Вывод метрик в консоль для проверки
    print("\nМетрики по группам:")
    print(report_df[['Accuracy', 'F1-Score', 'Mean_Value']])


if __name__ == "__main__":
    build_combined_report()