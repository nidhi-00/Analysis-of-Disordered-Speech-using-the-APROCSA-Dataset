# Analysis of Disordered Speech using the APROCSA Dataset

This repository contains the implementation for a computational pyscholinguistics project which analyzes disordered speech using the APROCSA clinical speech dataset. The project includes transcript preprocessing, statistical analysis, perceptual correlation, visualization, and regression modelling.

---

## Repository Structure

```
Analysis-of-Disordered-Speech-using-the-APROCSA-Dataset/

Dataset/                  # Raw CHAT transcripts
Outputs/
  ├── Dataset with PAR only/   # Participant-only transcripts
  ├── Statistics/              # CSV results
  └── Figures/                 # Plots
Scripts/                  # All Python scripts
A1.pdf                    # Assignment description
report.pdf                #Final results compiled into a report
```

---

## How to Run

Run all commands from the `Scripts/` folder.

### 1. Extract participant speech

```bash
python extract_par.py ../Dataset
```

### 2. Compute transcript statistics (A3)

```bash
python get_stats.py ../Dataset
```

### 3. Compute perceptual correlations (A4)

```bash
python compute_spearman.py
```

### 4. Generate scatter plots (A5)

```bash
python plot.py
```

### 5. Run regression model (Part B)

```bash
python partB_regression.py
```

---

## Outputs

* `Outputs/Statistics/a3_table.csv` – transcript features
* `Outputs/Statistics/perceptual_ratings.csv` – clinical ratings
* `Outputs/Figures/A5_scatter_plots.png` – visualization

---

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```


