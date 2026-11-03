import os

DATASET_FOLDER = "dataset"

total_images = 0

print("\n========================================")
print("       DATASET CHECK")
print("========================================")

for sign in sorted(os.listdir(DATASET_FOLDER)):

    sign_path = os.path.join(DATASET_FOLDER, sign)

    if not os.path.isdir(sign_path):
        continue

    normal_path = os.path.join(sign_path, "normal")
    dark_path = os.path.join(sign_path, "dark")

    normal_count = 0
    dark_count = 0

    if os.path.exists(normal_path):
        normal_count = len([
            f for f in os.listdir(normal_path)
            if f.lower().endswith(".jpg")
        ])

    if os.path.exists(dark_path):
        dark_count = len([
            f for f in os.listdir(dark_path)
            if f.lower().endswith(".jpg")
        ])

    total = normal_count + dark_count
    total_images += total

    print(
        f"{sign}: "
        f"Normal = {normal_count}, "
        f"Dark = {dark_count}, "
        f"Total = {total}"
    )

print("\n========================================")
print(f"TOTAL IMAGES: {total_images}")
print("========================================")