import re
import numpy as np
import torch
import csv
import os

MISSING_MARK = -1
VALID_BASES = {'A', 'T', 'C', 'G'}
CSV_OUTPUT_PATH = r"./data/2019/maize_genotype.csv"


def scan_vcf_metadata(vcf_file):
    sample_names = []
    valid_snp_ids = []

    with open(vcf_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#CHROM'):
                line_parts = line.split('\t')
                sample_names = line_parts[9:] if len(line_parts) > 9 else []
                sample_names = [s.split('_')[0] for s in sample_names]
                continue

            if line.startswith('##'):
                continue

            fields = line.split('\t')
            if len(fields) < 10:
                continue

            alt = fields[4].upper()
            if ',' in alt:
                continue

            ref = fields[3].upper()
            if ref not in VALID_BASES or alt not in VALID_BASES or ref == alt:
                continue

            chrom = fields[0]
            pos = fields[1]
            snp_id = fields[2] if fields[2] != '.' else f"{chrom}:{pos}"
            valid_snp_ids.append(snp_id)

    total_snps = len(valid_snp_ids)
    return sample_names, valid_snp_ids, total_snps

def parse_vcf_all_data(vcf_file):
    sample_names, valid_snp_ids, total_snps = scan_vcf_metadata(vcf_file)
    total_samples = len(sample_names)

    genotype_matrix = np.full((total_samples, total_snps), MISSING_MARK, dtype=np.int8)
    snp_idx = 0

    with open(vcf_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('##') or line.startswith('#CHROM'):
                continue

            if snp_idx >= total_snps:
                break

            fields = line.split('\t')
            if len(fields) < 10:
                continue

            ref = fields[3].upper()
            alt = fields[4].upper()

            if ',' in alt or ref not in VALID_BASES or alt not in VALID_BASES or ref == alt:
                continue

            genotypes = fields[9:] if len(fields) > 9 else []
            if len(genotypes) < total_samples:
                genotypes += ['./.'] * (total_samples - len(genotypes))

            for j, gt in enumerate(genotypes):
                if j >= total_samples:
                    break

                gt_field = gt.split(':')[0].strip()

                if gt_field in [".", "./.", ".|.", "*", ""]:
                    genotype_matrix[j, snp_idx] = MISSING_MARK
                    continue

                gt_parts = re.split(r'[/|]', gt_field)
                if len(gt_parts) != 2:
                    genotype_matrix[j, snp_idx] = MISSING_MARK
                    continue

                try:
                    a = int(gt_parts[0])
                    b = int(gt_parts[1])
                except (ValueError, TypeError):
                    genotype_matrix[j, snp_idx] = MISSING_MARK
                    continue

                if a == 0 and b == 0:
                    genotype_matrix[j, snp_idx] = 0
                elif a == 1 and b == 1:
                    genotype_matrix[j, snp_idx] = 1
                else:
                    genotype_matrix[j, snp_idx] = MISSING_MARK

            snp_idx += 1

    return genotype_matrix, sample_names, valid_snp_ids

def save_genotype_to_csv(genotype_matrix, sample_names, snp_ids, csv_path):
    csv_dir = os.path.dirname(csv_path)
    if not os.path.exists(csv_dir) and csv_dir != '':
        os.makedirs(csv_dir)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_f:
        csv_writer = csv.writer(csv_f)

        csv_header = ["Sample"] + snp_ids
        csv_writer.writerow(csv_header)

        for i, sample in enumerate(sample_names):
            sample_geno = genotype_matrix[i, :].tolist()
            csv_writer.writerow([sample] + sample_geno)



if __name__ == "__main__":
    VCF_FILE_PATH = r"./data/2019/maize_ld_pruned_final.vcf"

    geno_matrix, sample_names, snp_ids = parse_vcf_all_data(VCF_FILE_PATH)

    geno_tensor = torch.tensor(geno_matrix, dtype=torch.long)

    save_genotype_to_csv(geno_matrix, sample_names, snp_ids, CSV_OUTPUT_PATH)

    count_0 = torch.sum(geno_tensor == 0).item()
    count_1 = torch.sum(geno_tensor == 1).item()
    count_missing = torch.sum(geno_tensor == MISSING_MARK).item()
    total_count = geno_tensor.numel()

