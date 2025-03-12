import os
import csv
import json


def sum_last_column(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            total = 0
            with open(filepath, 'r') as file:
                reader = csv.reader(file, delimiter='\t')
                next(reader)  # Skip the header row
                for row in reader:
                    if row:
                        total += int(row[-1])  # Sum values from the last column
            results[filename] = total

    # Sort the dictionary by filename
    sorted_results = {filename: results[filename] for filename in sorted(results, key=lambda x: int(x.split('-')[1].split('.')[0]))}
    return sorted_results


def save_results_to_json(results, output_file):
    # Convert results dictionary to a list of dictionaries
    results_list = [{'args': k, 'mean_ms': v} for k, v in results.items()]
    with open(output_file, 'w') as file:
        json.dump(results_list, file, indent=4)


# Specify the directory containing your CSV files
directory_path = './profile-results'
sums = sum_last_column(directory_path)

output_json_file = 'log-archive/multistage-OCR.DEMAND_DATA_RD.L3_MISS.json'
save_results_to_json(sums, output_json_file)
