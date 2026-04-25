import random


def generate_psp():
    lengths = [50, 100, 500, 1_000, 5_000, 10_000, 50_000]

    print("\nФормат вывода:")
    print("1 - HEX (0a1f...)")
    print("2 - Bits (0101...)")
    fmt = input("Выбор: ")
    filename = "stego_payload_long.txt" if fmt == "1" else "stego_payload.bin"
    with open(filename, "w") as f:
        for size in lengths:
            data = bytes([random.getrandbits(8) for _ in range(size)])

            if fmt == "1":
                result = data.hex()
            else:
                result = "".join(f"{b:08b}" for b in data)
            
            f.write(result + "\n")
            print(f"Добавлено в {filename} ({len(result)} символов)")


if __name__ == "__main__":
    generate_psp()