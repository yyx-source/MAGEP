import pandas as pd
import os

target_dir = r"./data/2019/split_by_location"
keep_backup = False

for filename in os.listdir(target_dir):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(target_dir, filename)


    df = pd.read_csv(file_path, encoding='utf-8')

    df["Pedigree"] = df["Pedigree"].str.replace("/LH195", "", regex=False)

    if keep_backup:
        backup_path = f"{file_path}.bak"
        os.rename(file_path, backup_path)

    df.to_csv(file_path, index=False, encoding='utf-8')

