import os
import pandas as pd

ENV_DIR = r"/data/2019/split_by_environment"
BACKUP_RAW_FILE = False
TARGET_FILE_MARK = "ARH1_env"

def process_single_env_file(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")

    file_name = os.path.basename(file_path)
    is_target_file = TARGET_FILE_MARK in file_name

    mandatory_time_cols = ["Month", "Day", "Year", "Time"]
    keep_cols = [
        "Field Location",
        "Temperature [C]",
        "Relative Humidity [%]",
        "Solar Radiation [W/m2]",
        "Rainfall [mm]"
    ]

    missing_cols = []
    for col in mandatory_time_cols + keep_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        if is_target_file:
            prompt = f"Missing key columns {', '.join(missing_cols)}"
        else:
            prompt = f"Missing key columns {', '.join(missing_cols)}"
        print(prompt)
        return

    for col in mandatory_time_cols:
        df[col] = df[col].astype(str).fillna("0")

    df["Time_combined"] = (
            df["Year"].str.strip() + "-" +
            df["Month"].str.strip().str.zfill(2) + "-" +
            df["Day"].str.strip().str.zfill(2) + " " +
            df["Time"].str.strip()
    )

    final_cols = ["Field Location", "Time_combined"] + keep_cols[1:]
    final_df = df[final_cols].rename(columns={"Time_combined": "Time"})

    if BACKUP_RAW_FILE:
        file_dir = os.path.dirname(file_path)
        file_base, file_ext = os.path.splitext(file_name)
        backup_file_path = os.path.join(file_dir, f"{file_base}_backup{file_ext}")

        if not os.path.exists(backup_file_path):
            os.rename(file_path, backup_file_path)
            if is_target_file:
                prompt = f"Save：{os.path.basename(backup_file_path)}"
            else:
                prompt = f"Save：{os.path.basename(backup_file_path)}"
            print(prompt)

    final_df.to_csv(file_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    env_files = [
        f for f in os.listdir(ENV_DIR)
        if f.endswith(".csv")
    ]

    target_file_exists = any(TARGET_FILE_MARK in f for f in env_files)

    target_files = [f for f in env_files if TARGET_FILE_MARK in f]
    other_files = [f for f in env_files if TARGET_FILE_MARK not in f]

    for target_file in target_files:
        target_file_path = os.path.join(ENV_DIR, target_file)
        process_single_env_file(target_file_path)

    for env_file in other_files:
        env_file_path = os.path.join(ENV_DIR, env_file)
        process_single_env_file(env_file_path)

