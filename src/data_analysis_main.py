import os
from data_loader import DataLoader
from visualization_3_5 import run_violinplot_boxplot
from visualization_4_0 import run_errorbar_and_histogram_plots
from visualization_4_5 import run_correlation_heatmap
from visualization_5_0 import run_regression_plots
from visualization_5_5 import run_dimensionality_reduction_visualizations

def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Błąd ładowania danych.")
        return

    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_violinplot_boxplot(df, main_output_dir)
    run_errorbar_and_histogram_plots(df, main_output_dir)
    run_regression_plots(df, main_output_dir)
    run_correlation_heatmap(df, main_output_dir)
    run_dimensionality_reduction_visualizations(df, main_output_dir)

    print("\n Wszystkie wykresy zostały wygenerowane i zapisane w folderze 'plots'.")

if __name__ == '__main__':
    main()
