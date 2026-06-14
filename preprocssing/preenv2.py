import os
import pandas as pd

RAW_ENV_DIR = r"/data/2019/split_by_environment"
OUTPUT_ENV_DIR = r"/data/2019/environment"

TARGET_ENV_COLS = [
    "Temperature [C]",
    "Relative Humidity [%]",
    "Solar Radiation [W/m2]",
    "Rainfall [mm]"
]

def process_single_env_file(raw_file_path, output_dir):

    df = pd.read_csv(raw_file_path, encoding="utf-8")

    file_name = os.path.basename(raw_file_path)

    for col in TARGET_ENV_COLS:
        if col in df.columns:
            col_data = df[col].replace("", pd.NA)
            is_col_all_null = col_data.isna().all()

            if is_col_all_null:
                df[col] = 0
        else:
            df[col] = 0

    for col in TARGET_ENV_COLS:
        if col in df.columns:
            df[col] = df[col].replace("T", 0)
            df[col] = df[col].replace("t", 0)

    df = df.dropna(how="any").reset_index(drop=True)

    output_file_path = os.path.join(output_dir, file_name)
    df.to_csv(output_file_path, index=False, encoding="utf-8")


if __name__ == "__main__":
        if not os.path.exists(OUTPUT_ENV_DIR):
            os.makedirs(OUTPUT_ENV_DIR)

        raw_env_files = [
            f for f in os.listdir(RAW_ENV_DIR)
            if f.endswith(".csv")
        ]

        for raw_file in raw_env_files:
            raw_file_path = os.path.join(RAW_ENV_DIR, raw_file)
            process_single_env_file(raw_file_path, OUTPUT_ENV_DIR)
