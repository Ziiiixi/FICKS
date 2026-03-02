import csv

# File paths
# first_file_path = "/home/zixi/orion/profiling/benchmarks/restnet_bz32/resnet152_fwd_Grid_Block"  # The first file you mentioned
# csv_file_path = "/home/zixi/orion/profiling/benchmarks/restnet_bz32/knee_points_analysis_resnet.csv"   # The CSV file you mentioned
first_file_path = "/home/zixi/orion_bu/profiling/benchmarks/resnet152_bz32/resnet152_32_fwd"  # The first file you mentioned
csv_file_path = "/home/zixi/orion_bu/profiling/postprocessing/15percent/tpc_15percent_Resnet152_32.csv"   # The CSV file you mentioned
output_file_path = "updated_first_file"

# Load the data from the first file, including the header
first_file_data = []
with open(first_file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header
    first_file_data = list(reader)

# Add Knee_TPC to the header
header.append("Knee_TPC")

# Load the data from the CSV file and organize it by Kernel Name, Grid Size, and Block Size
csv_data = {}
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file) 
    print(reader.fieldnames)
    for row in reader:
        kernel_name = row['\ufeffKernel Name']
        grid_size = row['Grid Size']
        block_size = row['Block Size']
        knee_tpc = row['Knee_TPC']
        key = (kernel_name, grid_size, block_size)
        csv_data[key] = knee_tpc

# Add Knee_TPC to the first file data when Kernel Name, Grid Size, and Block Size match
for row in first_file_data:
    kernel_name = row[0]
    grid_size = row[5]  # Assuming Grid Size is the 6th column
    block_size = row[6] # Assuming Block Size is the 7th column
    key = (kernel_name, grid_size, block_size)
    if key in csv_data:
        row.append(csv_data[key])
    else:
        row.append('')  # Add an empty value if no match is found

# Save the updated data back to a new file, including the modified header
with open(output_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the updated header
    writer.writerows(first_file_data)

print("File updated successfully!")