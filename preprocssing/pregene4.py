import os
import pandas as pd

GENOTYPE_RAW_DIR = r"./data/2019/genotype"
PHENOTYPE_PROCESSED_DIR = r"./data/2019/phenotype"
GENOTYPE_OUTPUT_DIR = r"./data/2019/genetype1"

os.makedirs(GENOTYPE_OUTPUT_DIR, exist_ok=True)

for gene_filename in os.listdir(GENOTYPE_RAW_DIR):
    if gene_filename.endswith("_pheno.csv"):
        prefix = gene_filename.replace("_pheno.csv", "")
        pheno_filename = f"{prefix}_pheno.csv"

        pheno_path = os.path.join(PHENOTYPE_PROCESSED_DIR, pheno_filename)
        gene_path = os.path.join(GENOTYPE_RAW_DIR, gene_filename)
        output_path = os.path.join(GENOTYPE_OUTPUT_DIR, gene_filename)


        df_gene = pd.read_csv(gene_path, engine='python')
        df_gene_deduplicated = df_gene.drop_duplicates(subset="Sample", keep='first', ignore_index=False)

        df_pheno = pd.read_csv(pheno_path, engine='python')
        df_gene_aligned = df_gene_deduplicated.set_index("Sample").reindex(df_pheno["Sample"]).reset_index()

        df_gene_aligned.to_csv(output_path, index=False, encoding="utf-8")
