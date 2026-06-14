import os
import pandas as pd

GENOTYPE_CSV_PATH = r"./data/2019/maize_genotype.csv"
PHENOTYPE_DIR = r"./data/2019/phenotype"
OUTPUT_GENOTYPE_DIR = r"./data/2019/genotype"


def load_total_genotype(genotype_path):
    geno_df = pd.read_csv(genotype_path, encoding="utf-8")
    geno_df = geno_df.set_index("Sample", drop=False)
    return geno_df


def process_single_phenotype(pheno_file_path, total_geno_df, output_dir):
    pheno_df = pd.read_csv(pheno_file_path, encoding="utf-8")

    pheno_samples = pheno_df["Sample"].tolist()

    matched_geno_df = total_geno_df[total_geno_df["Sample"].isin(pheno_samples)]
    matched_geno_df = matched_geno_df.set_index("Sample").loc[pheno_samples].reset_index()

    pheno_filename = os.path.basename(pheno_file_path)
    output_geno_path = os.path.join(output_dir, pheno_filename)

    matched_geno_df.to_csv(output_geno_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_GENOTYPE_DIR):
        os.makedirs(OUTPUT_GENOTYPE_DIR)

    total_geno_df = load_total_genotype(GENOTYPE_CSV_PATH)

    pheno_files = [f for f in os.listdir(PHENOTYPE_DIR) if f.endswith(".csv")]

    for pheno_file in pheno_files:
        pheno_file_path = os.path.join(PHENOTYPE_DIR, pheno_file)
        process_single_phenotype(pheno_file_path, total_geno_df, OUTPUT_GENOTYPE_DIR)
