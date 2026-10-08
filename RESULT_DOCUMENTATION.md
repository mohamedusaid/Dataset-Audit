# TinyGPT Dataset Audit — Final Results

## 1. Executive Summary

This document records the completed audit of the TinyGPT dataset.

The dataset contains:

- **28 shards**
- **135,004,641 total tokens**
- **182,649 documents**
- **143,976 unique documents**
- **0 invalid token IDs**

The main issue identified is substantial exact duplication:

- **1,525 duplicate groups**
- **38,673 duplicate excess documents**
- **21.17% duplicate excess by document count**
- **12,145,497 duplicate-excess tokens**
- **9.01% duplicate-excess token fraction**

Exact train/validation overlap was also detected:

- **73 exact overlapping document hashes**
- **2.5034% of unique validation hashes**

The dataset is structurally valid, but it should be cleaned of exact duplicates and train/validation leakage before being used for another serious pretraining run.

---

## 2. Dataset Overview

| Metric | Result |
|---|---:|
| Shards | 28 |
| Total tokens | 135,004,641 |
| Train tokens | 132,307,294 |
| Validation tokens | 2,697,347 |
| Documents | 182,649 |
| Unique documents | 143,976 |
| Invalid token IDs | 0 |
| EOS tokens | 182,623 |

---

## 3. Token Integrity

All audited shards contained valid GPT-2 token IDs.

- Minimum token ID: **0**
- Maximum token ID: **50,256**
- Invalid token IDs: **0**

Therefore, no binary/token corruption was detected by the integrity checks.

---

## 4. Shard Structure

The dataset contains:

### Training

- 27 training shards
- 26 shards contain 5,000,000 tokens each
- Final training shard contains 2,307,294 tokens

### Validation

- 1 validation shard
- 2,697,347 tokens

All shards were successfully read during the audit.

### Shard results

| Split | Shard | Tokens | Min ID | Max ID | Invalid | EOS |
|---|---|---:|---:|---:|---:|---:|
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0000.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,652 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0001.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,760 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0002.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,713 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0003.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,568 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0004.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,723 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0005.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,767 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0006.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,886 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0007.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,739 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0008.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,823 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0009.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,703 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0010.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,716 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0011.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,865 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0012.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,909 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0013.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,736 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0014.bin` | 5,000,000 | 0 | 50,256 | 0 | 7,005 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0015.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,766 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0016.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,717 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0017.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,830 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0018.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,587 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0019.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,772 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0020.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,697 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0021.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,701 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0022.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,623 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0023.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,827 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0024.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,945 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0025.bin` | 5,000,000 | 0 | 50,256 | 0 | 6,825 |
| train | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/train/train_shard_0026.bin` | 2,307,294 | 0 | 50,256 | 0 | 3,116 |
| val | `/content/audit_results/_prepared/kaggle/working/tiny-gpt-500m/data_shards/val/val_shard_0000.bin` | 2,697,347 | 0 | 50,256 | 0 | 3,652 |

---

## 5. Document Statistics

### Training

| Metric | Value |
|---|---:|
| Documents | 178,997 |
| Tokens in documents | 132,128,323 |
| Mean length | 738.16 |
| Median | 551 |
| P90 | 1,351 |
| P95 | 1,794 |
| P99 | 4,290.08 |
| Maximum | 24,919 |

### Validation

| Metric | Value |
|---|---:|
| Documents | 3,652 |
| Tokens in documents | 2,693,695 |
| Mean length | 737.59 |
| Median | 547.5 |
| P90 | 1,351 |
| P95 | 1,782.35 |
| P99 | 4,589.77 |
| Maximum | 11,785 |

---

## 6. Exact Duplicate Analysis

The audit found:

- Documents hashed: **182,649**
- Unique documents: **143,976**
- Duplicate groups: **1,525**
- Duplicate excess documents: **38,673**
- Duplicate excess document fraction: **21.17%**

A separate exact token-mass analysis found:

- Duplicate excess tokens: **12,145,497**
- Duplicate token fraction: **9.01%**

The token percentage is the more useful measurement for estimating how much training-token budget is being consumed by exact duplicates.

### Major duplicate groups

Several documents occur thousands of times.

| Copies | Tokens/document |
|---:|---:|
| 9,385 | 270 |
| 9,385 | 364 |
| 9,193 | 276 |
| 9,193 | 261 |

These four groups alone account for approximately **10.89 million duplicate-excess tokens**.

This indicates that the duplication is highly concentrated rather than being evenly distributed across the dataset.

---

## 7. Train/Validation Leakage

Exact hash comparison identified:

- Train unique hashes: **141,133**
- Validation unique hashes: **2,916**
- Exact overlapping hashes: **73**
- Validation overlap fraction: **2.5034%**

Therefore, the validation set is not completely isolated from the training set.

Exact overlapping documents should be removed from the appropriate split before using the dataset for reliable evaluation.

---

## 8. Sampled Quality Heuristics

20,000 documents were sampled.

| Heuristic | Flagged |
|---|---:|
| Contains URL | 3.62% |
| Contains email | 0.98% |
| HTML-like | 17.14% |
| Repeated characters | 25.82% |
| Repeated punctuation | 0.11% |

These are **heuristic signals, not confirmed quality problems**.

In particular, programming content can naturally contain:

- angle brackets
- repeated characters
- formatting patterns
- HTML/XML
- code syntax

Therefore these flags must not be interpreted as automatic reasons to delete documents.

---

## 9. Final Findings

### Positive findings

- All 28 shards were readable.
- Token IDs are valid.
- No invalid token IDs were detected.
- Dataset structure is internally consistent.
- Document statistics were successfully calculated.

### Problems identified

#### 1. Exact duplication — HIGH PRIORITY

**9.01% of document tokens are duplicate-excess tokens.** This is the primary dataset problem.

#### 2. Train/validation leakage — HIGH PRIORITY

73 exact document hashes occur in both splits.

#### 3. Heuristic quality flags — INVESTIGATE, DO NOT AUTO-DELETE

The quality heuristics identify potentially suspicious material, but they are not sufficiently reliable for automatic removal.

---

## 10. Recommended Cleaning

Before future pretraining:

1. Remove exact duplicate copies while retaining one copy.
2. Remove exact train/validation overlaps.
3. Keep legitimate programming/code content.
4. Do not automatically remove documents solely because they trigger `html_like` or `repeated_characters`.
5. Re-audit the cleaned dataset.
6. Verify token counts and split separation again before training.

---

## 11. Final Verdict

### Dataset integrity

**PASS**

The tokenized dataset is structurally valid and contains no invalid token IDs.

### Dataset quality

**NEEDS CLEANING**

The dataset contains significant exact duplication and measurable train/validation leakage.

### Training readiness

**NOT RECOMMENDED AS-IS**

The dataset should be cleaned before being used for another serious pretraining run.

---

## 12. Important Limitation

This audit establishes exact duplication, exact train/validation overlap, token integrity, document statistics, and sampled heuristic signals.

It does **not** establish that every unique document is high quality. It also does not measure semantic/near-duplicate similarity across documents.

Therefore, the audit should not be interpreted as proof that the remaining unique dataset is completely clean.

---

## 13. Reproducibility

The audit was performed using the `Dataset-Audit` project.

The audit generated structured results including:

- shard statistics
- token statistics
- document statistics
- duplicate statistics
- leakage statistics
- quality statistics
- n-gram statistics
- samples
- Markdown/HTML reports

The original dataset remains separate from the audit source repository.

---

## 14. Final Status

**AUDIT COMPLETE**

The audit successfully identified the major structural and data-quality issues that need to be addressed before future training.

**Next stage: dataset cleaning and re-audit.**
