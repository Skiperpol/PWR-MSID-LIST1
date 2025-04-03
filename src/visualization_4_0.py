import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

def create_custom_error_bar_plot_with_hue(df, numeric_var, category_var, hue_var, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"error_bar_{numeric_var.replace(' ', '_')}_by_{category_var.replace(' ', '_')}_hue_{hue_var.replace(' ', '_')}.png"
    )
    
    for col in [numeric_var, category_var, hue_var]:
        if col not in df.columns:
            print(f"Brak kolumny: {col}")
            return

    df[category_var] = df[category_var].astype('category')
    df[hue_var] = df[hue_var].astype('category')
    
    cat_values = df[category_var].cat.categories
    hue_values = df[hue_var].cat.categories
    
    x_cat = np.arange(len(cat_values))
    n_hue = len(hue_values)
    if n_hue > 1:
        offsets = np.linspace(-0.3, 0.3, n_hue)
    else:
        offsets = [0]
    
    x_positions = {}
    for i, cat in enumerate(cat_values):
        for j, hue in enumerate(hue_values):
            x_positions[(cat, hue)] = x_cat[i] + offsets[j]
    
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, 
        sharex=True, 
        gridspec_kw={"height_ratios": [1, 3]},
        figsize=(10, 6)
    )
    
    for cat in cat_values:
        for hue in hue_values:
            group = df[(df[category_var]==cat) & (df[hue_var]==hue)][numeric_var].dropna()
            if len(group) == 0:
                continue
            mean_val = group.mean()
            sem_val = group.sem()
            x_pos = x_positions[(cat, hue)]
            color_idx = list(hue_values).index(hue)
            ax_top.errorbar(
                x_pos, 0, 
                xerr=sem_val, 
                fmt='o', 
                capsize=5, 
                capthick=2, 
                elinewidth=2, 
                color=f'C{color_idx}'
            )
            ax_top.text(x_pos, 0.1, f"{mean_val:.1f}", ha='center', va='bottom', fontsize=8)
    
    ax_top.set_ylim([-1, 1])
    ax_top.set_yticks([])
    ax_top.set_title(f"{numeric_var} – Mean ± SEM\npodział na {category_var} i {hue_var}")

    for cat in cat_values:
        for hue in hue_values:
            group = df[(df[category_var]==cat) & (df[hue_var]==hue)][numeric_var].dropna()
            if len(group) == 0:
                continue
            x_pos = x_positions[(cat, hue)]
            color_idx = list(hue_values).index(hue)
            ax_bottom.scatter(np.repeat(x_pos, len(group)), group, alpha=0.7, color=f'C{color_idx}')
    
    ax_bottom.set_xticks(x_cat)
    ax_bottom.set_xticklabels(cat_values, rotation=45)
    ax_bottom.set_ylabel(numeric_var)
    ax_bottom.set_title("Rozrzut poszczególnych obserwacji")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Custom error bar plot zapisano: {output_path}")

def create_histogram(df, numeric_var, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"histogram_{numeric_var.replace(' ', '_')}.png")
    
    if numeric_var not in df.columns:
        print(f"Brak kolumny: {numeric_var}")
        return

    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x=numeric_var, bins=30, kde=False)
    plt.title(f"Histogram {numeric_var}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram zapisano: {output_path}")

def create_histogram_with_hue(df, numeric_var, hue_var, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"histogram_{numeric_var.replace(' ', '_')}_hue_{hue_var.replace(' ', '_')}.png"
    )
    
    for col in [numeric_var, hue_var]:
        if col not in df.columns:
            print(f"Brak kolumny: {col}")
            return

    df[hue_var] = df[hue_var].astype('category')
    
    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x=numeric_var, hue=hue_var, bins=30, kde=False)
    plt.title(f"Histogram {numeric_var} (hue: {hue_var})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram z hue zapisano: {output_path}")

def run_errorbar_and_histogram_plots(df, main_output_dir):
    output_dir_error = os.path.join(main_output_dir, "error_bars")
    output_dir_hist = os.path.join(main_output_dir, "histograms")
    output_dir_hist_hue = os.path.join(main_output_dir, "histograms_with_hue")
    os.makedirs(output_dir_error, exist_ok=True)
    os.makedirs(output_dir_hist, exist_ok=True)
    os.makedirs(output_dir_hist_hue, exist_ok=True)

    error_bar_specs = [
        {"numeric_var": "Curricular units 1st sem (grade)", "category_var": "Gender", "hue_var": "Gender"},
        {"numeric_var": "Curricular units 1st sem (grade)", "category_var": "Target", "hue_var": "Target"},
        {"numeric_var": "Age at enrollment", "category_var": "Application mode", "hue_var": "Application mode"},
        {"numeric_var": "Admission grade", "category_var": "Marital status", "hue_var": "Marital status"},
        {"numeric_var": "Admission grade", "category_var": "Application mode", "hue_var": "Application mode"}
    ]

    for spec in error_bar_specs:
        create_custom_error_bar_plot_with_hue(
            df,
            numeric_var=spec["numeric_var"],
            category_var=spec["category_var"],
            hue_var=spec["hue_var"],
            output_dir=output_dir_error
        )

    histogram_specs = [
        {"numeric_var": "Unemployment rate"},
        {"numeric_var": "Age at enrollment"},
        {"numeric_var": "Curricular units 1st sem (credited)"},
        {"numeric_var": "Curricular units 1st sem (evaluations)"},
        {"numeric_var": "Curricular units 2nd sem (grade)"},
        {"numeric_var": "Inflation rate"},
        {"numeric_var": "GDP"}
    ]

    for spec in histogram_specs:
        create_histogram(
            df,
            numeric_var=spec["numeric_var"],
            output_dir=output_dir_hist
        )

    histogram_with_hue_specs = [
        {"numeric_var": "Curricular units 1st sem (credited)", "hue_var": "Target"},
        {"numeric_var": "Curricular units 1st sem (grade)", "hue_var": "Gender"},
        {"numeric_var": "Admission grade", "hue_var": "Application mode"},
        {"numeric_var": "Age at enrollment", "hue_var": "Scholarship holder"},
        {"numeric_var": "Curricular units 2nd sem (evaluations)", "hue_var": "Daytime/evening attendance"},
        {"numeric_var": "Curricular units 2nd sem (grade)", "hue_var": "Scholarship holder"}
    ]
    
    for spec in histogram_with_hue_specs:
        create_histogram_with_hue(
            df,
            numeric_var=spec["numeric_var"],
            hue_var=spec["hue_var"],
            output_dir=output_dir_hist_hue
        )



def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Błąd ładowania danych.")
        return

    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_errorbar_and_histogram_plots(df, main_output_dir)
    print("\nWszystkie wykresy zostały wygenerowane.")

if __name__ == '__main__':
    main()
