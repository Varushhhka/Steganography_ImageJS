import os
import math
from PIL import Image


def get_channel_matrix(img, channel_index):
    pixels = list(img.getdata())
    width, height = img.size

    matrix = []
    row = []

    for i, px in enumerate(pixels):
        row.append(px[channel_index])

        if (i + 1) % width == 0:
            matrix.append(row)
            row = []

    return matrix

def discrimination(group):
    s = 0

    rows = len(group)
    cols = len(group[0])

    for r in range(rows):
        for c in range(cols - 1):
            s += abs(group[r][c] - group[r][c + 1])

    for c in range(cols):
        for r in range(rows - 1):
            s += abs(group[r][c] - group[r + 1][c])

    return s

def apply_mask(group, mask):
    out = []

    for r in range(len(group)):
        row = []

        for c in range(len(group[0])):
            val = group[r][c]

            if val % 2 == 0:
                val += mask[r][c]
            else:
                val -= mask[r][c]

            row.append(val)

        out.append(row)

    return out

def flip_group(group):
    out = []

    for row in group:
        line = []

        for val in row:
            if val % 2 == 0:
                line.append(val + 1)
            else:
                line.append(val - 1)

        out.append(line)

    return out

def cut_group(img, y, x, h, w):
    group = []

    for r in range(h):
        row = []

        for c in range(w):
            row.append(img[y + r][x + c])

        group.append(row)

    return group

def rs_channel(matrix, mask):
    h = len(matrix)
    w = len(matrix[0])

    mh = len(mask)
    mw = len(mask[0])

    neg_mask = []

    for row in mask:
        neg_mask.append([-v for v in row])

    Rm = Sm = 0
    Rn = Sn = 0
    RFm = SFm = 0
    RFn = SFn = 0

    groups = 0

    for y in range(0, h - mh + 1, mh):
        for x in range(0, w - mw + 1, mw):

            G = cut_group(matrix, y, x, mh, mw)
            F = flip_group(G)

            dG = discrimination(G)

            dGm = discrimination(apply_mask(G, mask))
            dGn = discrimination(apply_mask(G, neg_mask))

            dF = discrimination(F)

            dFm = discrimination(apply_mask(F, mask))
            dFn = discrimination(apply_mask(F, neg_mask))

            if dGm > dG:
                Rm += 1
            elif dGm < dG:
                Sm += 1

            if dGn > dG:
                Rn += 1
            elif dGn < dG:
                Sn += 1

            if dFm > dF:
                RFm += 1
            elif dFm < dF:
                SFm += 1

            if dFn > dF:
                RFn += 1
            elif dFn < dF:
                SFn += 1

            groups += 1

    if groups == 0:
        return 0

    d0 = (Rm - Sm) / groups
    dn0 = (Rn - Sn) / groups
    d1 = (RFm - SFm) / groups
    dn1 = (RFn - SFn) / groups

    a = 2 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3 * d0
    c = d0 - dn0

    if a == 0:
        return 0

    disc = b * b - 4 * a * c

    if disc < 0:
        return 0

    x1 = (-b + math.sqrt(disc)) / (2 * a)
    x2 = (-b - math.sqrt(disc)) / (2 * a)

    x = x1 if abs(x1) < abs(x2) else x2

    if abs(x - 0.5) < 1e-9:
        return 0

    p = abs(x / (x - 0.5))

    if p < 0:
        p = 0
    if p > 1:
        p = 1

    return p

def analyze_image(path, mask):
    try:
        img = Image.open(path).convert("RGB")
    except:
        return 0, 0

    width, height = img.size
    total_pixels = width * height
    max_bytes = (total_pixels * 3) // 8

    r = get_channel_matrix(img, 0)
    g = get_channel_matrix(img, 1)
    b = get_channel_matrix(img, 2)

    pr = rs_channel(r, mask)
    pg = rs_channel(g, mask)
    pb = rs_channel(b, mask)

    bits_r = pr * total_pixels
    bits_g = pg * total_pixels
    bits_b = pb * total_pixels

    total_bits = bits_r + bits_g + bits_b
    total_bytes = int(total_bits / 8)

    percent = (total_bytes / max_bytes * 100) if max_bytes > 0 else 0

    return total_bytes, percent

def category(size_bytes):
    if size_bytes < 100:
        return "<100"
    elif size_bytes < 500:
        return "100-500"
    elif size_bytes < 1000:
        return "500-1000"
    elif size_bytes < 10000:
        return "1000-10000"
    elif size_bytes < 20000:
        return "10000-20000"
    elif size_bytes < 30000:
        return "20000-30000"
    elif size_bytes < 40000:
        return "30000-40000"
    elif size_bytes < 50000:
        return "40000-50000"
    else:
        return ">=50000"


def process_folder(folder, mask_choice=1):
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

    if mask_choice == 1:
        mask = [[0, 1, 0]]
    elif mask_choice == 2:
        mask = [[0, 1],
                [1, 0]]
    else:
        mask = [[0, 1, 0]]

    print(f"Размер маски: {len(mask)}×{len(mask[0])}\n")

    stats = {
        "<100": 0,
        "100-500": 0,
        "500-1000": 0,
        "1000-10000": 0,
        "10000-20000": 0,
        "20000-30000": 0,
        "30000-40000": 0,
        "40000-50000": 0,
        ">=50000": 0
    }

    percent_stats = {
        "<1%": 0,
        "1-5%": 0,
        "5-10%": 0,
        "10-20%": 0,
        "20-30%": 0,
        "30-40%": 0,
        "40-50%": 0,
        ">50%": 0
    }

    print("Первые 100 файлов:\n")
    print(f"{'№':<4} {'Файл':<40} {'Байт':<12} {'% от макс':<12} {'Категория'}")
    print("-" * 85)

    for i, file in enumerate(files):
        path = os.path.join(folder, file)

        size_bytes, percent = analyze_image(path, mask)
        cat = category(size_bytes)

        stats[cat] += 1

        if percent < 1:
            percent_cat = "<1%"
        elif percent < 5:
            percent_cat = "1-5%"
        elif percent < 10:
            percent_cat = "5-10%"
        elif percent < 20:
            percent_cat = "10-20%"
        elif percent < 30:
            percent_cat = "20-30%"
        elif percent < 40:
            percent_cat = "30-40%"
        elif percent < 50:
            percent_cat = "40-50%"
        else:
            percent_cat = ">50%"

        percent_stats[percent_cat] += 1

        if i < 100:
            print(f"{i + 1:<4} {file:<40} {size_bytes:<12} {percent:<11.2f}% {cat}")

        if (i + 1) % 50 == 0:
            print(f"\nОбработано {i + 1}/{len(files)}\n")

    print("\n" + "=" * 85)
    print("ИТОГ")
    print("=" * 85)

    print("\n--- РАСПРЕДЕЛЕНИЕ ПО РАЗМЕРУ (байты) ---")
    print(f"<100 bytes        : {stats['<100']}")
    print(f"100-500 bytes     : {stats['100-500']}")
    print(f"500-1000 bytes    : {stats['500-1000']}")
    print(f"1000-10000 bytes  : {stats['1000-10000']}")
    print(f"10000-20000 bytes : {stats['10000-20000']}")
    print(f"20000-30000 bytes : {stats['20000-30000']}")
    print(f"30000-40000 bytes : {stats['30000-40000']}")
    print(f"40000-50000 bytes : {stats['40000-50000']}")
    print(f">=50000 bytes     : {stats['>=50000']}")

    print("\n--- РАСПРЕДЕЛЕНИЕ ПО ПРОЦЕНТУ ЗАПОЛНЕНИЯ (от макс. ёмкости) ---")
    for cat, cnt in percent_stats.items():
        pct = 100 * cnt / max(len(files), 1)
        print(f"{cat:>10}: {cnt:4d} ({pct:5.1f}%)")

    print(f"\nВсего файлов: {len(files)}")


if __name__ == "__main__":
    folder = input("Введите путь к папке PNG: ").strip()
    print("\nВыберите маску (make your choice):")
    print("1. Маска 1×3 [0,1,0]")
    print("2. Маска 2×2 [[0,1],[1,0]]")
    mask_choice = input("Выбор маски(1 или 2, по умолчанию 1): ")

    if mask_choice == "2":
        mask_choice = 2
    else:
        mask_choice = 1

    process_folder(folder, mask_choice)