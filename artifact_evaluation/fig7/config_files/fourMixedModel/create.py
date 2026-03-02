import json
import itertools

# Models list
models = [
    "Rnet_8", "Mnet_32", "R1net_8", "Vnet_8", "Dnet_8"
]

# Function to create configuration based on net type and number
def create_config(net, number):
    # Kernel counts based on the specified values
    kernel_counts = {
        "Mnet_32": 1354,
        "Rnet_8": 2340,
        "R1net_8": 1575,  # ResNet101
        "Vnet_8": 296,    # VGG19
        "Dnet_8": 3298,    # DenseNet
        "Tnet_8": 152,    # Transformer
        "Bnet_8": 286,    # Transformer
    }

    # Convert the batch size to an integer
    batch_size = int(number)

    if "Mnet" in net:
        return {
            "arch": "mobilenet_v2",
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/mobilenetv2_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "mobilenet_v2",
                "batchsize": batch_size,  # Ensure batchsize is an integer
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Rnet" in net:
        return {
            "arch": "resnet152",  # Updated to resnet152
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/resnet152_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "resnet152",  # Updated to resnet152
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "R1net" in net:
        return {
            "arch": "resnet101",  # Updated to ResNet101
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/resnet101_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "resnet101",
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Vnet" in net:
        return {
            "arch": "vgg19",  # Updated to VGG19
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/vgg19_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "vgg19",
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Dnet" in net:
        return {
            "arch": "densenet201",  # Updated to DenseNet
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/densenet201_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "densenet201",
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Tnet" in net:
        return {
            "arch": "vit_b_16",  # Updated to Transformer
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/vitb16_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "vit_b_16",
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }
    elif "Bnet" in net:
        return {
            "arch": "bert",  # Updated to Transformer
            "kernel_file": f"/home/zixi/orion_bu/benchmarking/model_kernels/small_models/KRISP/bert_{batch_size}_fwd_KRISP",
            "num_kernels": kernel_counts[net],
            "num_iters": 20,
            "args": {
                "model_name": "bert",
                "batchsize": batch_size,
                "rps": 20,
                "uniform": False,
                "dummy_data": True,
                "train": False
            }
        }

# Generate all combinations of two models
combinations = list(itertools.combinations(models, 4))

# Loop through the combinations and create JSON files
for entry in combinations:
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
    
        # Output the message in the requested format
    print(f'    # ("", "", "{entry[0]}_{entry[1]}", 160000),')
    
    # print(f'Created file: {file_name}')
