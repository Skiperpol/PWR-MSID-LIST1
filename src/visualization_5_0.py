import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import DataLoader

def create_lmplot(df, x, y, hue, title, x_label, y_label, output_path):
    if hue:
        g = sns.lmplot(x=x, y=y, hue=hue, data=df, ci=95, 
                       height=6, aspect=1.2, scatter_kws={"s": 50, "alpha": 0.7})
    else:
        g = sns.lmplot(x=x, y=y, data=df, ci=95, 
                       height=6, aspect=1.2, scatter_kws={"s": 50, "alpha": 0.7})
    
    g.set_axis_labels(x_label, y_label)
    plt.title(title)
    plt.tight_layout()
    g.savefig(output_path)
    plt.close()
    print(f"Wykres regresji zapisano: {output_path}")

def run_regression_plots(df, main_output_dir):
    regression_dir = os.path.join(main_output_dir, "regression")
    os.makedirs(regression_dir, exist_ok=True)

    regression_specs = [
        {
            "x": "Curricular units 1st sem (credited)",
            "y": "Curricular units 1st sem (enrolled)",
            "hue": None,
            "title": "Regresja: Jednostki (credited) vs. Jednostki (enrolled)",
            "x_label": "Curricular units 1st sem (credited)",
            "y_label": "Curricular units 1st sem (enrolled)",
            "output_filename": "regression_credited_enrolled.png"
        },
        {
            "x": "Age at enrollment",
            "y": "Curricular units 1st sem (grade)",
            "hue": None,
            "title": "Regresja: Wiek przy zapisach vs. Ocena (1 sem.)",
            "x_label": "Age at enrollment",
            "y_label": "Curricular units 1st sem (grade)",
            "output_filename": "regression_age_grade.png"
        },
        {
            "x": "Admission grade",
            "y": "Previous qualification (grade)",
            "hue": None,
            "title": "Regresja: Wyniki rekrutacji vs. Ocena z poprzedniej kwalifikacji",
            "x_label": "Admission grade",
            "y_label": "Previous qualification (grade)",
            "output_filename": "regression_admission_previous.png"
        },
        {
            "x": "Inflation rate",
            "y": "GDP",
            "hue": None,
            "title": "Regresja: Inflacja vs. PKB",
            "x_label": "Inflation rate",
            "y_label": "GDP",
            "output_filename": "regression_inflation_gdp.png"
        },
        {
            "x": "Unemployment rate",
            "y": "GDP",
            "hue": None,
            "title": "Regresja: Bezrobocie vs. PKB",
            "x_label": "Unemployment rate",
            "y_label": "GDP",
            "output_filename": "regression_unemployment_gdp.png"
        },
        {
            "x": "Curricular units 2nd sem (grade)",
            "y": "Curricular units 2nd sem (approved)",
            "hue": None,
            "title": "Regresja: Ocena (2 sem.) vs. Zaliczenia (2 sem.)",
            "x_label": "Curricular units 2nd sem (grade)",
            "y_label": "Curricular units 2nd sem (approved)",
            "output_filename": "regression_units2_grade_approved.png"
        },
        {
            "x": "Age at enrollment",
            "y": "Admission grade",
            "hue": None,
            "title": "Regresja: Wiek przy zapisach vs. Wyniki rekrutacyjne",
            "x_label": "Age at enrollment",
            "y_label": "Admission grade",
            "output_filename": "regression_age_admission.png"
        }
    ]

    for spec in regression_specs:
        output_path = os.path.join(regression_dir, spec["output_filename"])
        create_lmplot(
            df,
            x=spec["x"],
            y=spec["y"],
            hue=spec["hue"],
            title=spec["title"],
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            output_path=output_path
        )

def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Błąd przy wczytywaniu danych.")
        return

    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_regression_plots(df, main_output_dir)
    print("\nWszystkie wykresy zostały wygenerowane.")
    
if __name__ == '__main__':
    main()
