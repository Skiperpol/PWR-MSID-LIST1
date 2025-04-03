import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath, sep=';')
            return df
        except Exception as e:
            print(f"Błąd ładowania danych: {e}")
            return None

if __name__ == '__main__':
    loader = DataLoader("data.csv")
    df = loader.load_data()
    if df is not None:
        print(df.head())
