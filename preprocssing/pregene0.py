import os
import subprocess
import re

plink_path = r"F:\Software\plink_win64_20241022\plink.exe"
input_vcf = r"./data/2019/G2F.vcf"
fixed_vcf = r"./data/2019/G2F_fixed.vcf"
basic_qc_prefix = r"./data/2019/maize_basic_qc"
ld_prune_prefix = r"./data/2019/maize_ld_pruned"

GENO_THRESHOLD = 0.05
MIND_THRESHOLD = 0.1

LD_WINDOW = 50
LD_STEP = 10
LD_R2 = 0.1

for prefix in [basic_qc_prefix, ld_prune_prefix]:
    output_dir = os.path.dirname(prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

basic_qc_cmd = [
    plink_path,
    "--vcf", fixed_vcf,
    "--mind", str(MIND_THRESHOLD),
    "--geno", str(GENO_THRESHOLD),
    "--allow-extra-chr",
    "--set-missing-var-ids", "@:#",
    "--make-bed",
    "--recode", "vcf-iid",
    "--out", basic_qc_prefix
]


basic_result = subprocess.run(
    basic_qc_cmd,
    check=True,
    capture_output=True,
    text=True
)

basic_match = re.search(r'(\d+) variants and (\d+) people pass filters and QC',
                        basic_result.stdout + basic_result.stderr)
basic_snps = int(basic_match.group(1)) if basic_match else 0
basic_people = int(basic_match.group(2)) if basic_match else 0

basic_bed_file = f"{basic_qc_prefix}.bed"
basic_vcf_file = f"{basic_qc_prefix}.vcf"


ld_prune_cmd = [
    plink_path,
    "--bfile", basic_qc_prefix,
    "--indep-pairwise", str(LD_WINDOW), str(LD_STEP), str(LD_R2),
    "--allow-extra-chr",
    "--out", ld_prune_prefix
]

ld_filter_prefix = f"{ld_prune_prefix}_final"
ld_filter_cmd = [
    plink_path,
    "--bfile", basic_qc_prefix,
    "--extract", f"{ld_prune_prefix}.prune.in",
    "--allow-extra-chr",
    "--make-bed",
    "--recode", "vcf-iid",
    "--out", ld_filter_prefix
]


if __name__ == "__main__":
    subprocess.run(ld_prune_cmd, check=True, capture_output=True, text=True)
    ld_filter_result = subprocess.run(
        ld_filter_cmd,
        check=True,
        capture_output=True,
        text=True
    )

    ld_match = re.search(r'(\d+) variants and (\d+) people pass filters and QC',
                         ld_filter_result.stdout + ld_filter_result.stderr)
    ld_final_snps = int(ld_match.group(1)) if ld_match else 0
    ld_final_people = int(ld_match.group(2)) if ld_match else 0

    prune_in_count = sum(1 for _ in open(f"{ld_prune_prefix}.prune.in", 'r', encoding='utf-8')) if os.path.exists(
        f"{ld_prune_prefix}.prune.in") else 0
    prune_out_count = sum(1 for _ in open(f"{ld_prune_prefix}.prune.out", 'r', encoding='utf-8')) if os.path.exists(
        f"{ld_prune_prefix}.prune.out") else 0

    ld_final_vcf = f"{ld_filter_prefix}.vcf"
    ld_vcf_snp_count = 0
    if os.path.exists(ld_final_vcf):
        ld_vcf_snp_count = sum(1 for line in open(ld_final_vcf, 'r', encoding='utf-8') if not line.startswith('#'))


