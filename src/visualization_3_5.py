import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader

PLOTS_TO_CREATE = [
    ("Target", "Age at enrollment"),
    ("Gender", "Curricular units 1st sem (grade)"),
    ("Gender", "Curricular units 1st sem (credited)"),
    ("Target", "Curricular units 1st sem (grade)"),
    ("Application mode", "Age at enrollment"),
    ("Marital status", "Admission grade"),
    ("Previous qualification", "Curricular units 1st sem (evaluations)"),
    ("Application mode", "Admission grade"),
    ("Scholarship holder", "Curricular units 2nd sem (grade)"),
    ("Course", "Curricular units 1st sem (evaluations)"),
    ("Gender", "Admission grade"),
]

HUE_TRANSLATIONS = {
    "gender": {1: "Mężczyzna", 0: "Kobieta"},
    "international": {1: "Międzynarodowy", 0: "Lokalny"},
    "tuition fees up to date": {1: "Opłacone", 0: "Nieopłacone"},
    "debtor": {1: "Dłużnik", 0: "Bez długu"},
    "displaced": {1: "Zamiejscowy", 0: "Niezamiejscowy"},
    "mother's qualification": {1: "Średnie", 2: "Licencjat", 3: "Inżynier", 4: "Magister", 5: "Doktorat"},
    "previous qualification": {1: "Średnie", 2: "Licencjat", 3: "Inżynier", 4: "Magister", 5: "Doktorat"}
}

COLUMN_LABELS = {
    "daytime/evening attendance": "Tryb nauczania",
    "curricular units 1st sem (grade)": "Średnia ocena (1 semestr)",
    "gender": "Płeć",
    "marital status": "Stan cywilny",
    "age at enrollment": "Wiek przy zapisach",
    "international": "Międzynarodowość",
    "scholarship holder": "Stypendysta",
    "curricular units 2nd sem (approved)": "Zaliczone jednostki (2 semestr)",
    "application mode": "Tryb aplikacji",
    "curricular units 1st sem (without evaluations)": "Jednostki bez ocen (1 semestr)",
    "tuition fees up to date": "Czesne",
    "previous qualification": "Poprzednie kwalifikacje",
    "curricular units 2nd sem (evaluations)": "Oceny cząstkowe (2 semestr)",
    "target": "Grupa docelowa",
    "debtor": "Dłużnik",
    "curricular units 2nd sem (grade)": "Średnia ocena (2 semestr)",
    "displaced": "Status zamiejscowy",
    "curricular units 1st sem (approved)": "Zaliczone jednostki (1 semestr)",
    "mother's qualification": "Wykształcenie matki"
}

def translate_label(col_name):
    return COLUMN_LABELS.get(col_name.strip().lower(), col_name)

def get_matching_column(df, col_name):
    for c in df.columns:
        if c.strip().lower() == col_name.strip().lower():
            return c
    return None

def translate_hue_column(df, hue_col):
    mapping = HUE_TRANSLATIONS.get(hue_col.strip().lower())
    if mapping:
        df = df.copy()
        df[hue_col] = df[hue_col].map(mapping)
    return df

def create_boxplot(df, x, y, hue, title, output_path):
    actual_x = get_matching_column(df, x)
    actual_y = get_matching_column(df, y)
    actual_hue = get_matching_column(df, hue)
    if actual_x is None or actual_y is None or actual_hue is None:
        print(f"Brak kolumn: {x}, {y} lub {hue}.")
        return
    df = translate_hue_column(df, actual_hue)
    plt.figure(figsize=(8,6))
    sns.boxplot(x=actual_x, y=actual_y, hue=actual_hue, data=df)
    plt.title(title)
    plt.xlabel(translate_label(actual_x))
    plt.ylabel(translate_label(actual_y))
    plt.legend(title=translate_label(actual_hue))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Boxplot zapisano: {output_path}")

def create_violinplot(df, x, y, hue, title, output_path):
    actual_x = get_matching_column(df, x)
    actual_y = get_matching_column(df, y)
    actual_hue = get_matching_column(df, hue)
    if actual_x is None or actual_y is None or actual_hue is None:
        print(f"Brak kolumn: {x}, {y} lub {hue}.")
        return
    df = translate_hue_column(df, actual_hue)
    plt.figure(figsize=(8,6))
    sns.violinplot(x=actual_x, y=actual_y, hue=actual_hue, data=df)
    plt.title(title)
    plt.xlabel(translate_label(actual_x))
    plt.ylabel(translate_label(actual_y))
    plt.legend(title=translate_label(actual_hue))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Violinplot zapisano: {output_path}")

def run_violinplot_boxplot(df, main_output_dir):
    boxplot_dir = os.path.join(main_output_dir, "boxplots")
    violinplot_dir = os.path.join(main_output_dir, "violinplots")

    os.makedirs(boxplot_dir, exist_ok=True)
    os.makedirs(violinplot_dir, exist_ok=True)

    plots_to_create = PLOTS_TO_CREATE

    for cat_col, num_col in plots_to_create:
        title = f"{translate_label(num_col)} wg {translate_label(cat_col)}"
        output_path_box = os.path.join(boxplot_dir, f"boxplot_{num_col.replace(' ', '_')}_by_{cat_col.replace(' ', '_')}.png")
        output_path_violin = os.path.join(violinplot_dir, f"violinplot_{num_col.replace(' ', '_')}_by_{cat_col.replace(' ', '_')}.png")
        
        create_boxplot(df, x=cat_col, y=num_col, hue=cat_col, title=title, output_path=output_path_box)
        create_violinplot(df, x=cat_col, y=num_col, hue=cat_col, title=title, output_path=output_path_violin)



def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Błąd ładowania danych.")
        return
    
    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_violinplot_boxplot(df, main_output_dir)
    print("\nWszystkie wykresy zostały wygenerowane.")

if __name__ == '__main__':
    main()
