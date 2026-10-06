import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# SECTION A: Data Loading
# ==========================================
# Load the raw dataset into a pandas DataFrame
df = pd.read_csv('Fundamentals_of_Data_Mining_Milestone1_Raw_Dataset.csv')

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Last 5 Rows ---")
print(df.tail())

# ==========================================
# SECTION B: Data Inspection
# ==========================================
print("\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Duplicate Records Count ---")
print(df.duplicated().sum())

print("\n--- Basic Statistical Summary ---")
print(df.describe())

# ==========================================
# SECTION C: Data Cleaning
# ==========================================
# 1. Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# 2. Standardize text values (capitalize/title case Gender)
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()

# 3. Handle anomalies/outliers in numerical columns
# Attendance valid percentage range is 0 to 100
df.loc[df['Attendance'] > 100, 'Attendance'] = None
df.loc[df['Attendance'] < 0, 'Attendance'] = None

# 4. Handle missing numerical values using median imputation
num_cols = ['Age', 'Attendance', 'Quiz_Score', 'Assignment_Score']
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("\n--- Verification After Cleaning ---")
print("Missing values remaining:", df.isnull().sum().sum())
print("Duplicates remaining:", df.duplicated().sum())

# ==========================================
# SECTION D: Data Transformation
# ==========================================
# Create Average_Score column from the score/metric columns
df['Average_Score'] = df[['Quiz_Score', 'Assignment_Score']].mean(axis=1)

# Categorize student performance
def categorize_score(score):
    if score >= 90:
        return 'Excellent'
    elif score >= 80:
        return 'Good'
    else:
        return 'Needs Improvement'

df['Score_Category'] = df['Average_Score'].apply(categorize_score)

# Save cleaned and transformed dataset
df.to_csv('cleaned_dataset.csv', index=False)

# ==========================================
# SECTION E: Basic Exploratory Data Analysis (EDA)
# ==========================================
overall_avg = df['Average_Score'].mean()
highest_avg = df['Average_Score'].max()
lowest_avg = df['Average_Score'].min()
category_counts = df['Score_Category'].value_counts()

# Course with the highest average score
course_avg = df.groupby('Course')['Average_Score'].mean()
top_course = course_avg.idxmax()

print("\n--- EDA RESULTS ---")
print(f"Overall Average Score: {overall_avg:.2f}")
print(f"Highest Average Score: {highest_avg:.2f}")
print(f"Lowest Average Score: {lowest_avg:.2f}")
print("\nStudents per Category:")
print(category_counts)
print(f"\nCourse with Highest Average Score: {top_course} ({course_avg[top_course]:.2f})")

# Visualization: Score Category Distribution
plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color=['#2ca02c', '#1f77b4', '#d62728'])
plt.title('Distribution of Student Performance Categories')
plt.xlabel('Score Category')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('screenshots/data_analysis.png') # Saves plot to file
plt.show()

# ==========================================
# SECTION VI: Reflection
# ==========================================
"""
REFLECTION:
Data preprocessing is a vital foundation in the data mining process because raw datasets often contain 
missing values, duplicate entries, inconsistent entries, and extreme anomalies that can negatively skew statistical analysis. 
By cleaning and transforming the raw data, we ensure high data quality, leading to accurate, reliable, 
and trustworthy insights when applying data mining algorithms. Ultimately, effective preprocessing prevents 
"garbage in, garbage out" scenarios, improving overall decision-making and performance evaluation.
"""
