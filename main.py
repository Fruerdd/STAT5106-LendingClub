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
    # Common LendingClub data leakage fields (these are post-origination info; using them in a predictive model is cheating)
    leakage_columns = [
        'recoveries', 'collection_recovery_fee', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'out_prncp', 'out_prncp_inv',
        'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 'issue_d'
    ]

    # Check if our data contains any of these "cheating" fields
    found_leakage = [col for col in leakage_columns if col in df.columns]

    if found_leakage:
        print(
            f"⚠️ Warning: Potential data leakage features found!\nThese features contain post-origination information and should be removed before modeling:\n{found_leakage}")
    else:
        print("✅ Great! The cleaning code was robust, and no obvious common data leakage features were found.")


if __name__ == "__main__":
    # Point to the cleaned data file you just generated
    cleaned_data_path = "data/cleaned/lending_club_clean.csv.gz"
    validate_and_check_data(cleaned_data_path)




