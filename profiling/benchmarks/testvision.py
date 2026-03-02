import time
import torch
import torchvision
from torchvision import models
from torchviz import make_dot  # Import torchviz

print(torchvision.__file__)

def vision(model_name, batchsize, local_rank, do_eval=True, profile=None):

    # Prepare data and target
    data = torch.ones([batchsize, 3, 224, 224], pin_memory=True).to(local_rank)
    target = torch.ones([batchsize], pin_memory=True).to(torch.long).to(local_rank)

    # Load model
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(local_rank)

    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(local_rank)

    # Run only one iteration
    start = time.time()
    if profile == 'ncu':
        torch.cuda.nvtx.range_push("start")
    elif profile == 'nsys':
        torch.cuda.profiler.cudart().cudaProfilerStart()

    # with torch.no_grad():
        output = model(data)

    # Generate and save the computational graph
    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.render("model_computational_graph", format="png")
    print("Computational graph saved as 'model_computational_graph.png'.")

    print(f"Iteration took {time.time()-start} sec")

if __name__ == "__main__":
    vision('mobilenet_v2', 32, 0, True, 'nsys')
