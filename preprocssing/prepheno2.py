import pandas as pd
import os

compare_txt = r"./data/2019/compare.txt"
source_dir = r"./data/2019/split_by_location"
output_dir = r"./data/2019/phenotype"
file_encoding = "utf-8"

def load_compare_mapping(txt_path):
    pedigree_to_sample = {}
    pedigree_order = []

    with open(txt_path, 'r', encoding=file_encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            sample = parts[0]
            pedigree = parts[1]
            pedigree_to_sample[pedigree] = sample
            pedigree_order.append(pedigree)
    return pedigree_to_sample, pedigree_order

pedigree_to_sample, pedigree_order = load_compare_mapping(compare_txt)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(source_dir):
    if not filename.endswith(".csv"):
        continue

    csv_path = os.path.join(source_dir, filename)
    df = pd.read_csv(csv_path, encoding=file_encoding)

    df_filtered = df[df["Pedigree"].isin(pedigree_order)].copy()

    df_filtered["Pedigree"] = pd.Categorical(
        df_filtered["Pedigree"],
        categories=pedigree_order,
        ordered=True
    )
    df_sorted = df_filtered.sort_values("Pedigree").reset_index(drop=True)

    df_sorted["Sample"] = df_sorted["Pedigree"].map(pedigree_to_sample)

    pedigree_col_idx = df_sorted.columns.get_loc("Pedigree")
    cols = list(df_sorted.columns)
    cols.remove("Sample")
    cols.insert(pedigree_col_idx + 1, "Sample")
    df_sorted = df_sorted[cols]

    output_path = os.path.join(output_dir, filename)
    df_sorted.to_csv(output_path, index=False, encoding=file_encoding)

