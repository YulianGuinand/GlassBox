import os
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris, load_wine

def generate_samples():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Génération des données dans '{output_dir}/'...")
    
    # 1. Housing (Régression)
    print("   - california_housing.csv (Régression Immobilière)...")
    housing = fetch_california_housing(as_frame=True)
    df_housing = housing.frame
    df_housing.to_csv(f"{output_dir}/california_housing.csv", index=False)
    
    # 2. Iris (Classification Simple)
    print("   - iris.csv (Classification Fleurs)...")
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    # Add some random dates to test date handling
    df_iris['date_observation'] = pd.date_range(start='2023-01-01', periods=len(df_iris))
    df_iris.to_csv(f"{output_dir}/iris.csv", index=False)
    
    # 3. Wine (Classification)
    print("   - wine.csv (Classification Vin)...")
    wine = load_wine(as_frame=True)
    df_wine = wine.frame
    df_wine.to_csv(f"{output_dir}/wine.csv", index=False)
    
    # 4. Titanic (Synthetic/Manual for variety)
    # Creating a small synthetic titanic-like dataset because fetch_openml can be slow/problematic without API key sometimes
    print("   - titanic_mini.csv (Classification Survie)...")
    data = {
        'Survived': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0] * 5,
        'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 2, 3] * 5,
        'Sex': ['male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male'] * 5,
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0, 27.0, 14.0, 4.0, 58.0, 20.0] * 5,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 11.13, 8.46, 16.7, 26.55, 8.05] * 5,
        'Embark_Date': pd.date_range(start='1912-04-10', periods=50) # Fake date for testing
    }
    df_titanic = pd.DataFrame(data)
    df_titanic.to_csv(f"{output_dir}/titanic_mini.csv", index=False)

    print("✅ Terminé !")

if __name__ == "__main__":
    generate_samples()
