import pandas as pd
import os

input_csv = r"./data/2019/2019_weather_cleaned.csv"
output_dir = "./data/2019/split_by_environment"

df = pd.read_csv(input_csv, encoding='utf-8')
location_col = df.columns[0]
unique_locations = df[location_col].unique()
for location in unique_locations:
    df_location = df[df[location_col] == location]
    output_filename = f"{location}_env.csv"
    output_path = os.path.join(output_dir, output_filename)
    df_location.to_csv(output_path, index=False, encoding='utf-8')
