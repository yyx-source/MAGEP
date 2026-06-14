input_vcf = r"./data/2019/maize_ld_pruned_final.vcf"
preview_lines = 50

def preview_raw_vcf(vcf_path, line_count=50):
    with open(vcf_path, 'r', encoding='utf-8', errors='ignore') as vcf_f:
        for line_num, line in enumerate(vcf_f, 1):
            if line_num > line_count:
                break

            line_content = line.rstrip('\n')

            if line.startswith('##'):
                print(f"{line_num:2d}：{line_content}")
            elif line.startswith('#CHROM'):
                print(f"{line_num:2d}]：{line_content}")
            elif line.strip() == '':
                print(f"{line_num:2d}")
            else:
                print(f"{line_num:2d}：{line_content}")


if __name__ == "__main__":
    preview_raw_vcf(input_vcf, preview_lines)

