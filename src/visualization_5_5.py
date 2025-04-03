import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
from enums import NUMERIC_COLUMNS

def preprocess_data(df):
    """
    1. Wybieramy wyłącznie kolumny określone w NUMERIC_COLUMNS (tylko te, które występują w danych).
    2. Wypełniamy ewentualne brakujące wartości średnią.
    3. Standaryzujemy dane (przekształcamy je tak, by miały średnią 0 i odchylenie standardowe 1).
    """
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    df_numeric = df[available_columns].copy()
    
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    return scaled_data, available_columns

def visualize_pca(data, df, output_path):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    
    if "Target" in df.columns:
        pca_df["Target"] = df["Target"]
    else:
        pca_df["Target"] = "All"
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Target", palette="Set1", s=50, alpha=0.7)
    plt.title("PCA - 2 Component Visualization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"PCA visualization saved to: {output_path}")

def visualize_tsne(data, df, output_path, random_state=42):
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_components = tsne.fit_transform(data)
    tsne_df = pd.DataFrame(data=tsne_components, columns=["TSNE1", "TSNE2"])
    
    if "Target" in df.columns:
        tsne_df["Target"] = df["Target"]
    else:
        tsne_df["Target"] = "All"
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="Target", palette="Set1", s=50, alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"t-SNE visualization saved to: {output_path}")

def run_dimensionality_reduction_visualizations(df, main_output_dir):
    processed_data, used_columns = preprocess_data(df)

    output_dir = os.path.join(main_output_dir, "dimensionality_reduction")
    os.makedirs(output_dir, exist_ok=True)
    
    pca_output = os.path.join(output_dir, "pca_visualization.png")
    visualize_pca(processed_data, df, pca_output)
    
    tsne_output = os.path.join(output_dir, "tsne_visualization.png")
    visualize_tsne(processed_data, df, tsne_output)

def main():
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is None:
        print("Nie udało się wczytać danych.")
        return
    
    main_output_dir = "plots"
    os.makedirs(main_output_dir, exist_ok=True)

    run_dimensionality_reduction_visualizations(df, main_output_dir)
    print("\nWszystkie wykresy zostały wygenerowane.")

if __name__ == '__main__':
    main()
