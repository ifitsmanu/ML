import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def load_and_process_data(filenames):
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        df = clean_job_titles(df)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    combined_df = combined_df[['jobs_name', 'jobs_description', 'jobs_min_compensation', 'jobs_max_compensation', 'count_apps_id']]

    # Remove rows with 'NA' values in the compensation columns
    combined_df.dropna(subset=['jobs_min_compensation', 'jobs_max_compensation'], inplace=True)

    print(combined_df)
    return combined_df


def clean_job_titles(df):
    cna_pattern = r'\b(CNA|HHA|Caregiver|Nursing Assistant|Certified Nursing Assistant)\b'
    maid_pattern = r'\b(Maid|Housekeeper|Housekeeping)\b'
    cashier_pattern = r'\b(Cashier|Checkout|POS|Point of Sale)\b'
    cleaner_pattern = r'\b(Cleaner|Cleaning|Janitor|Janitorial)\b'

    def standardized_job_title(job_title):
        if re.search(cna_pattern, job_title, re.IGNORECASE):
            return 'CNA'
        elif re.search(maid_pattern, job_title, re.IGNORECASE):
            return 'Maid'
        elif re.search(cashier_pattern, job_title, re.IGNORECASE):
            return 'Cashier'
        elif re.search(cleaner_pattern, job_title, re.IGNORECASE):
            return 'Cleaner'
        else:
            return 'Other'

    df['jobs_name'] = df['jobs_name'].apply(standardized_job_title)
    return df

def plot_job_title_vs_applicants(df):
    job_titles = df['jobs_name'].unique()
    average_applicants = []

    for job_title in job_titles:
        avg_applicants = df[df['jobs_name'] == job_title]['count_apps_id'].mean()
        average_applicants.append(avg_applicants)

    plt.bar(job_titles, average_applicants)
    plt.xlabel('Job Title')
    plt.ylabel('Average Number of Applicants')
    plt.title('Average Number of Applicants per Job Title')
    plt.show()

def plot_description_length_vs_applicants(df):
    df['description_length'] = df['jobs_description'].str.len()
    plt.scatter(df['description_length'], df['count_apps_id'])
    plt.xlabel('Description Length')
    plt.ylabel('Number of Applicants')
    plt.title('Description Length vs. Number of Applicants')
    plt.show()

def plot_compensation_vs_applicants(df):
    plt.scatter(df['jobs_min_compensation'], df['count_apps_id'], label='Minimum Compensation')
    plt.scatter(df['jobs_max_compensation'], df['count_apps_id'], label='Maximum Compensation', alpha=0.1)
    plt.xlabel('Compensation')
    plt.ylabel('Number of Applicants')
    plt.title('Compensation vs. Number of Applicants')
    plt.legend()
    plt.xlim(0, 150000)  
    plt.show()

def main():
    filenames = ['cashier.csv', 'cleaner.csv', 'cna.csv', 'maid.csv']
    df = load_and_process_data(filenames)

    plot_job_title_vs_applicants(df)
    plot_description_length_vs_applicants(df)
    plot_compensation_vs_applicants(df)

if __name__ == '__main__':
    main()
