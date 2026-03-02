import csv
import re

# Define the paths for your log file and the existing contention matrix CSV file
log_file_path = '123.log'
csv_file_path = 'contention_matrix.csv'

# Step 1: Read the current contention matrix from the CSV file into a dictionary
contention_matrix = {}
with open(csv_file_path, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)  # Skip the header
    for row in reader:
        group_1, group_2, contention_factor = int(row[0]), int(row[1]), float(row[2])
        contention_matrix[(group_1, group_2)] = contention_factor
        contention_matrix[(group_2, group_1)] = contention_factor  # Since the matrix is symmetric

# Step 2: Parse the log file to extract contention factors
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        # Extract the contention factor, cluster pair, and duration from the line
        match = re.search(r"er (\d+) and cluster (\d+) is ([\d\.]+)", line)
        if match:
            cluster_1 = int(match.group(1))
            cluster_2 = int(match.group(2))
            contention_factor = float(match.group(3))

            # Update the corresponding entry in the contention matrix
            if (cluster_1, cluster_2) in contention_matrix:
                # Update the existing contention factor if necessary
                contention_matrix[(cluster_1, cluster_2)] = contention_factor
                contention_matrix[(cluster_2, cluster_1)] = contention_factor  # Ensure symmetry

# Step 3: Write the updated contention matrix back to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Group_1', 'Group_2', 'Contention_Factor'])

    # Write the updated matrix, maintaining the order
    for (group_1, group_2), contention_factor in sorted(contention_matrix.items()):
        writer.writerow([group_1, group_2, contention_factor])

print(f'Contention matrix CSV file "{csv_file_path}" has been updated successfully.')
