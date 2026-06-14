import os
import pandas as pd

RAW_PHENO_DIR = r"./data/2019/phenotype"
OUTPUT_PHENO_DIR = r"./data/2019/phenotype1"

TARGET_KEEP_COLS = [
    "Field-Location",
    "Sample",
    "Plant Height [cm]",
    "Grain Yield (bu/A)"
]

def process_single_pheno_file(raw_file_path, output_dir):
    df = pd.read_csv(raw_file_path, encoding="utf-8")

    file_name = os.path.basename(raw_file_path)
    original_rows = len(df)

    existing_target_cols = [col for col in TARGET_KEEP_COLS if col in df.columns]
    missing_target_cols = [col for col in TARGET_KEEP_COLS if col not in df.columns]

    df_keep = df[existing_target_cols].copy()
    for missing_col in missing_target_cols:
        df_keep[missing_col] = pd.NA

    df_keep = df_keep[TARGET_KEEP_COLS]

    df_clean = df_keep.dropna(how="any").reset_index(drop=True)

    output_file_path = os.path.join(output_dir, file_name)
    df_clean.to_csv(output_file_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PHENO_DIR):
        os.makedirs(OUTPUT_PHENO_DIR)

    raw_pheno_files = [
        f for f in os.listdir(RAW_PHENO_DIR)
        if f.endswith(".csv")
    ]

    for raw_file in raw_pheno_files:
        raw_file_path = os.path.join(RAW_PHENO_DIR, raw_file)
        process_single_pheno_file(raw_file_path, OUTPUT_PHENO_DIR)
