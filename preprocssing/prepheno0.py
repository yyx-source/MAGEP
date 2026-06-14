import pandas as pd
import os

input_csv = r"./data/2019/g2f_2019_phenotypic_clean_data.csv"
output_dir = "./data/2019/split_by_location"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(input_csv, encoding='utf-8')
location_col = df.columns[0]
unique_locations = df[location_col].unique()

for location in unique_locations:
    df_location = df[df[location_col] == location]
    output_filename = f"{location}_pheno.csv"
    output_path = os.path.join(output_dir, output_filename)
    df_location.to_csv(output_path, index=False, encoding='utf-8')

