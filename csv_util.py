import csv
import sys

csv.field_size_limit(sys.maxsize)

input_file = 'test_data.csv'
output_file = 'output1.csv'
headers = ['count_apps_id', 'job_id', 'job_title', 'job_description', 'min_pay', 'max_pay', 'commission', 'state', 'zip_code']

with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write headers to the output file
    writer.writerow(headers)

    # Write the existing data to the output file
    for row in reader:
        writer.writerow(row)
