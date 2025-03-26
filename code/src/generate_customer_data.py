import pandas as pd
import numpy as np
import random

# Define the number of rows to generate
num_rows = 10000

# Define possible values for categorical features
marital_statuses = ["Single", "Married", "Divorced"]
employment_statuses = ["Employed", "Self-Employed", "Unemployed"]
education_levels = ["High School", "Bachelor's", "Master's"]
home_ownerships = ["Own", "Rent", "Mortgage"]
loan_purposes = ["Home Loan", "Education", "Personal", "Auto Loan"]
states = ["AZ", "CA", "FL"]

# Generate synthetic data
data = {
    "age": np.random.randint(25, 59, size=num_rows),  # Age between 25 and 65
    "income": np.random.randint(30000, 150000, size=num_rows),  # Income between $30k and $150k
    #"credit_score": np.random.randint(620, 850, size=num_rows),  # Credit score between 620 and 850
    "account_balance": np.random.randint(0, 100000, size=num_rows),  # Account balance between $0 and $100k
    "existing_loans": np.random.randint(0, 5, size=num_rows),  # Number of existing loans (0-4)
    "marital_status": np.random.choice(marital_statuses, size=num_rows),  # Random marital status
    "employment_status": np.random.choice(employment_statuses, size=num_rows),  # Random employment status
    "education_level": np.random.choice(education_levels, size=num_rows),  # Random education level
    "debt_to_income_ratio": np.random.uniform(10, 50, size=num_rows).round(2),  # DTI ratio between 10% and 50%
    "home_ownership": np.random.choice(home_ownerships, size=num_rows),  # Random home ownership status
    "employment_duration": np.random.randint(0, 30, size=num_rows),  # Employment duration in years (0-30)
    "loan_purpose": np.random.choice(loan_purposes, size=num_rows),  # Random loan purpose
    "state": np.random.choice(states, size=num_rows),  # Random US state
    "dependents": np.random.randint(0, 5, size=num_rows),  # Number of dependents (0-4)
    "annual_expenses": np.random.randint(20000, 100000, size=num_rows),  # Annual expenses between $20k and $100k
}


# Add logic to determine if the customer applied for a home loan
def determine_applied_for_home_loan(row):
    # Higher likelihood for middle-aged customers with good credit and income
    if (
        30 <= row["age"] <= 50 and
        #row["credit_score"] >= 680 and
        row["income"] >= 60000 and
        row["account_balance"] >= 15000 and
        row["existing_loans"] <= 1 and
        row["debt_to_income_ratio"] <= 40 and
        row["employment_status"] in ["Employed", "Self-Employed"]
    ):
        return 1  # Likely to apply
    else:
        return 0  # Not likely to apply

# Apply the logic to generate the target column
df = pd.DataFrame(data)
df["applied_for_home_loan"] = df.apply(determine_applied_for_home_loan, axis=1)

# Ensure at least 30% of the rows have applied_for_home_loan = 1
while df["applied_for_home_loan"].mean() < 0.3:
    # Identify rows where applied_for_home_loan is 0
    rows_to_update = df[df["applied_for_home_loan"] == 0].sample(frac=0.1).index
    
    # Update these rows to meet the conditions for applied_for_home_loan = 1
    df.loc[rows_to_update, "age"] = np.random.randint(30, 51, size=len(rows_to_update))
    df.loc[rows_to_update, "existing_loans"] = np.random.randint(0,2, size=len(rows_to_update))
    #df.loc[rows_to_update, "credit_score"] = np.random.randint(680, 850, size=len(rows_to_update))
    df.loc[rows_to_update, "account_balance"] = np.random.randint(15000, 100000, size=len(rows_to_update))
    df.loc[rows_to_update, "income"] = np.random.randint(60000, 150000, size=len(rows_to_update))
    #df.loc[rows_to_update, "debt_to_income_ratio"] = np.random.uniform(10, 40, size=len(rows_to_update)).round(2)
    df.loc[rows_to_update, "employment_status"] = np.random.choice(["Employed", "Self-Employed"], size=len(rows_to_update))
    #marital_status
    df.loc[rows_to_update, "marital_status"] = np.random.choice(["Single", "Married"], size=len(rows_to_update))
    # Update employment_duration based on age
    df.loc[rows_to_update, "employment_duration"] = df.loc[rows_to_update, "age"].apply(
        lambda age: np.random.randint(2, max(0, age - 18) + 1)
    )
    # Ensure expense_to_income_ratio is below 50%
    df.loc[rows_to_update, "annual_expenses"] = (
        df.loc[rows_to_update, "income"] * np.random.uniform(0.1, 0.5, size=len(rows_to_update))
    ).astype(int)

    df.loc[rows_to_update, "debt_to_income_ratio"] = (
        (df.loc[rows_to_update, "annual_expenses"] / df.loc[rows_to_update, "income"]) * 100
    ).round(2)
    df.loc[rows_to_update, "home_ownership"] = np.random.choice(["Rent", "Own"], size=len(rows_to_update))
    #df.loc[rows_to_update, "loan_purpose"] = "Home Loan"
    
    # Recalculate applied_for_home_loan for these rows
    df.loc[rows_to_update, "applied_for_home_loan"] = df.loc[rows_to_update].apply(determine_applied_for_home_loan, axis=1)

# Verify the proportion of customers who applied for a home loan
print(f"Percentage of customers who applied for a home loan: {df['applied_for_home_loan'].mean() * 100:.2f}%")

# Save the dataset to a CSV file
output_file = "resources/customer_data_generated.csv"
df.to_csv(output_file, index=False)

print(f"Synthetic dataset with {num_rows} rows has been generated and saved to {output_file}.")