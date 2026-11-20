import pandas as pd


def validate_and_check_data(file_path):
    print("Reading cleaned data. The file is large, please wait...")
    # Pandas is smart enough to read .csv.gz compressed files directly without manual extraction
    df = pd.read_csv(file_path)

    print("\n" + "=" * 50)
    print("✅ 1. Pipeline Validation")
    print("=" * 50)
    print(f"Current row count: {df.shape[0]} rows")
    print(f"Current feature count: {df.shape[1]} columns")

    print("\n" + "=" * 50)
    print("🔍 2. Data Quality Check")
    print("=" * 50)
    # Check for missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if missing_data.empty:
        print("Excellent! There are no missing values in the cleaned data.")
    else:
        print("Note: The following features still contain missing values (showing counts):")
        print(missing_data.head(10))  # Only show the top 10 columns with the most missing values
        print("...")

    print("\n" + "=" * 50)
    print("🚨 3. Data Leakage Check")
    print("=" * 50)
    # Use the canonical post-outcome list; issue_d is an origination field, not leakage.
    from src.lending_club_data_cleaning import LEAKAGE_FIELDS

    found_leakage = sorted(LEAKAGE_FIELDS.intersection(df.columns))

    if found_leakage:
        print(
            f"⚠️ Warning: Potential data leakage features found!\nThese features contain post-origination information and should be removed before modeling:\n{found_leakage}")
    else:
        print("✅ Great! The cleaning code was robust, and no obvious common data leakage features were found.")


if __name__ == "__main__":
    cleaned_data_path = "data/cleaned/clean_v1/lending_club_clean.csv.gz"
    validate_and_check_data(cleaned_data_path)




