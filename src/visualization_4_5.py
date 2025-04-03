import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import DataLoader
from enums import NUMERIC_COLUMNS

def create_correlation_heatmap(df, output_path):
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if not available_columns:
        print("Brak dostępnych kolumn numerycznych do analizy.")
        return

    df_numeric = df[available_columns].copy()
    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmapa korelacji została zapisana w pliku: {output_path}")

def run_correlation_heatmap(df, main_output_dir):
    heatmap_dir = os.path.join(main_output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    output_path = os.path.join(heatmap_dir, "correlation_heatmap.png")
    create_correlation_heatmap(df, output_path)

def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Nie udało się wczytać danych.")
        return

    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_correlation_heatmap(df, main_output_dir)
    print("\nWszystkie wykresy zostały wygenerowane.")

if __name__ == '__main__':
    main()
