================================================================================
  MAGEP -- A meta-learning with adversarial domain adaptation method for
      few-shot cross-region maize height and yield joint prediction
================================================================================

1. OVERVIEW
--------------------------------------------------------------------------------

MAGEP predicts crop phenotypes (plant height and grain yield) from genotype
(SNP) and environment (weather time-series + statistics) data across multiple
field locations.

The model encodes genotype via dilated CNN, environment via Transformer + MLP,
fuses them through a gating interaction module, and predicts two phenotypic
traits simultaneously. A Gradient Reversal Layer (GRL) with a domain classifier
and CORAL/MMD loss align source/target feature distributions.

2. PROJECT STRUCTURE
--------------------------------------------------------------------------------

  MAGEP/
  |-- test_main.py               Main entry: training + fine-tuning + evaluation
  |-- preprocssing/              Data preprocessing scripts
  |   |-- preenv0.py            Split environment CSV by location
  |   |-- preenv1.py           Timestamp formatting
  |   |-- preenv2.py           Clean + fill missing environment data
  |   |-- pregene0.py          VCF QC + LD pruning (PLINK)
  |   |-- pregene1.py         VCF -> CSV genotype matrix
  |   |-- pregene2.py         Raw VCF preview utility
  |   |-- pregene3.py         Split genotype by location
  |   |-- pregene4.py         Deduplicate + align genotype with phenotype
  |   |-- prepheno0.py          Split phenotype CSV by location
  |   |-- prepheno1.py         Clean pedigree column
  |   |-- prepheno2.py         Column filtering (height + yield only)
  |   |-- prepheno3.py         Remove NaN rows
  |
  |-- model/
  |   |-- model.py               MAGEP Model
  |   |-- dataset.py             Dataset class (GEPDataset)
  |   |-- utils.py               Loss functions, Reptile-based
  |   |-- evalution.py           Evaluation metrics
  |
  |-- test_data/                 Processed data directory
  |   |-- genotype/              *_gene.csv  (one per location)
  |   |-- environment/            *_env.csv   (one per location)
  |   |-- phenotype/             *_pheno.csv (one per location)
  |
  |-- save_model/                Saved trained model
  |-- save_result/               Saved evaluation metrics


3. MODEL INPUTS
--------------------------------------------------------------------------------
The GEPDataset class (model/dataset.py) loads per-location CSV files and
produces the following tensors for each sample:
  G_seq    -- Genotype sequence
  E_stats  -- Environment summary statistics
  E_seq    -- Environment time-series
  y_norm   -- Normalized phenotype targets
  y_raw    -- Raw phenotype values (used for evaluation)

4. HOW TO RUN
--------------------------------------------------------------------------------
Prerequisites
 - Python 3.8+
 - PyTorch (CUDA recommended)
 - scipy, numpy, pandas, scikit-learn, matplotlib, tqdm
 - PLINK (for genotype QC preprocessing only)

Data Preprocessing
  Place raw data files and edit file paths in preprocessing scripts as needed.
  Run preprocessing scripts in the following order:

Training
  Run the pipeline:
    python test_main.py

Output
  After training completes, the following files are generated:
    {save_dir}/
    source_reptile_pretrain.pth   Stage 1 checkpoint
    best_da_model.pth             Best model from adversarial domain adaptation fine-tuning

    {save_result}/
    {name}_preds.csv              Per-sample predictions (raw + predicted)
    metrics.csv                   Summary metrics (RMSE, MAE, PCC)
