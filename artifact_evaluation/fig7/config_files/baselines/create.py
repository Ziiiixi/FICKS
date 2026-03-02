import json

# Provided configurations
data = [
    ("Rnet_8", "Mnet_8", "Rnet_16", "Mnet_16"),
    ("Mnet_8", "Rnet_32", "Mnet_16", "Rnet_8"),
    ("Rnet_32", "Mnet_32", "Rnet_8", "Mnet_16"),
    ("Mnet_16", "Rnet_16", "Mnet_32", "Rnet_32"),
    ("Rnet_8", "Mnet_32", "Rnet_32", "Mnet_16"),
    ("Mnet_32", "Rnet_8", "Mnet_8", "Rnet_16"),
    ("Mnet_16", "Mnet_8", "Rnet_32", "Rnet_16"),
    ("Rnet_8", "Rnet_8", "Mnet_32", "Mnet_16")
]

# Function to create configuration based on net type and number
def create_config(net, number):
    # Kernel counts based on the specified values
    kernel_counts = {
        "Mnet_8": 490,
        "Mnet_16": 778,
        "Mnet_32": 1354,
        "Rnet_8": 2340,
        "Rnet_16": 4012,
        "Rnet_32": 7356
    }

    # Convert the batch size to an integer
    batch_size = int(number)

    if "Mnet" in net:
        return {
            "arch": "mobilenet_v2",
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/15percent/mobilenetv2_{batch_size}_fwd_15percent",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "mobilenet_v2",
                "batchsize": batch_size,  # Ensure batchsize is an integer
                "rps": 40,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Rnet" in net:
        return {
            "arch": "resnet152",  # Updated to resnet152
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/15percent/resnet152_{batch_size}_fwd_15percent",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "resnet152",  # Updated to resnet152
                "batchsize": batch_size,  # Ensure batchsize is an integer
                "rps": 40,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }

# Loop through the data and create JSON files
for entry in data:
    # Create the file name by joining the elements with underscores
    file_name = "_".join(entry) + ".json"
    
    # Create the list of configurations for each entry
    config_list = []
    for name in entry:
        net, number = name.split('_')
        config_list.append(create_config(name, number))
    
    # Write the parsed data to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(config_list, json_file, indent=4)
    
    print(f'Created file: {file_name}')
