import os
import torch
import warnings


def setup_cuda(use_memory_fraction: float = 0.1, num_threads: int = 8, device: str = 'cuda',
               multiGPU: bool = False, visible_devices: str = "0", use_cuda_with_id: int = 0) -> None:

    if device == 'cpu':
        import multiprocessing
        print(f'Torch version: {torch.__version__}')
        print(f"Available CPU cores: {multiprocessing.cpu_count()}")
        torch.set_num_threads(num_threads)
        print(f"Torch using {num_threads} threads.")
        
    elif device == 'gpu' or device == "cuda":
        # setup environmental variables
        print(f'Torch version: {torch.__version__}')
        if int(torch.__version__.split("+")[0].split('.')[1]) >= 12:
            warnings.warn('Since 1.12.0+ torch has lazy init and may ignore environmental variables :( - consider variable: use_cuda_with_id')

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        assert torch.cuda.is_available(
        ), f"cuda is not available, cannot setup cuda devices: {visible_devices}"

        # Setup num threads limit - e.g. for data loader
        torch.set_num_threads(num_threads)
        print(f"Torch using {num_threads}/{torch.get_num_threads()} threads.")

        # check available
        devices = [d for d in range(torch.cuda.device_count())]
        device_names = [torch.cuda.get_device_name(d) for d in devices]
        device_to_name = dict(zip(devices, device_names))
        print(f"Available devices: {device_to_name}")

        # setup cuda
        if len(devices) > 1 and multiGPU:
            for dev_id in devices:
                cu_dev = torch.device(device, dev_id)
                print(f"***Device - {cu_dev} setup:***")
                gpu_properties = torch.cuda.get_device_properties(dev_id)
                total_memory = gpu_properties.total_memory
                memory_info = torch.cuda.mem_get_info()
                print(f"\tCuda available: {torch.cuda.is_available()}, device: {cu_dev} num. devices: {torch.cuda.device_count()}, device name: {gpu_properties.name}, free memory: {memory_info[0]/1024**2:.0f} MB, total memory: {total_memory/1024**2:.0f} MB.")
                torch.cuda.set_per_process_memory_fraction(use_memory_fraction, torch.cuda.current_device())
                print(f"\tMemory fraction: {use_memory_fraction}, memory limit {int(total_memory * use_memory_fraction)/1024**3:.2f} GB.")
            print(f"Setup completed - devices in use: {', '.join([f'{device}:{id}' for id in devices])}.")
        else:
            cu_dev = torch.device(device, use_cuda_with_id)
            print(f"***Device - {cu_dev} setup:***")
            gpu_properties = torch.cuda.get_device_properties(cu_dev)
            total_memory = gpu_properties.total_memory
            memory_info = torch.cuda.mem_get_info()
            print(f"\tCuda available: {torch.cuda.is_available()}, device: {cu_dev} num. devices: {torch.cuda.device_count()}, device name: {gpu_properties.name}, free memory: {memory_info[0]/1024**2:.0f} MB, total memory: {total_memory/1024**2:.0f} MB.")
            torch.cuda.set_per_process_memory_fraction(use_memory_fraction, cu_dev)
            print(f"\tMemory fraction: {use_memory_fraction}, memory limit {int(total_memory * use_memory_fraction)/1024**3:.2f} GB.")
            current_device_index = torch.cuda.current_device()
            print(f"Setup completed - device in use: {torch.device(device, current_device_index)}.")
        print('\n')


if __name__ == "__main__":

    # CONFIG EXAMPLES

    # # 0. simple dev for debugging - using device with id:0, visible 1 GPU
    setup_cuda(use_memory_fraction=0.2, visible_devices="0")
    # # Torch using 8 threads.
    # # Available devices: {0: 'NVIDIA A100 80GB PCIe'}
    # # ***Device - cuda:0 setup:***
    # #         Cuda available: True, device: cuda:0 num. devices: 1, device name: NVIDIA A100 80GB PCIe, free memory: 80222 MB, total memory: 81070 MB.
    # #         Memory fraction: 0.2, memory limit 15.83 GB.
    # # Setup completed - device in use: cuda:0.


    # # 1. training on gpu id:0 with more memory and more threads
    # setup_cuda(use_memory_fraction=0.75, num_threads=16, visible_devices="0")
    # # Torch using 16 threads.
    # # Available devices: {0: 'NVIDIA A100 80GB PCIe'}
    # # ***Device - cuda:0 setup:***
    # #         Cuda available: True, device: cuda:0 num. devices: 1, device name: NVIDIA A100 80GB PCIe, free memory: 80222 MB, total memory: 81070 MB.
    # #         Memory fraction: 0.75, memory limit 59.38 GB.
    # # Setup completed - device in use: cuda:0.

    # # 2. heavy training on second GPU with id:1
    # setup_cuda(use_memory_fraction=1.0, num_threads=20, visible_devices="0,1", use_cuda_with_id=1)
    # # OUTPUT:
    # # Torch using 20 threads.
    # # Available devices: {0: 'NVIDIA A100 80GB PCIe', 1: 'NVIDIA A100 80GB PCIe'}
    # # ***Device - cuda:1 setup:***
    # #         Cuda available: True, device: cuda:1 num. devices: 2, device name: NVIDIA A100 80GB PCIe, free memory: 80222 MB, total memory: 81070 MB.
    # #         Memory fraction: 1.0, memory limit 79.17 GB.
    # # Setup completed - device in use: cuda:1.


    # # 3. data parallel training on 2 GPUS
    # setup_cuda(use_memory_fraction=0.75, visible_devices="0,1", multiGPU=True)
    # # OUTPUT:
    # # Torch using 8 threads.
    # # Available devices: {0: 'NVIDIA A100 80GB PCIe', 1: 'NVIDIA A100 80GB PCIe'}
    # # ***Device - cuda:0 setup:***
    # #         Cuda available: True, device: cuda:0 num. devices: 2, device name: NVIDIA A100 80GB PCIe, free memory: 80222 MB, total memory: 81070 MB.
    # #         Memory fraction: 0.75, memory limit 59.38 GB.
    # # ***Device - cuda:1 setup:***
    # #         Cuda available: True, device: cuda:1 num. devices: 2, device name: NVIDIA A100 80GB PCIe, free memory: 80222 MB, total memory: 81070 MB.
    # #         Memory fraction: 0.75, memory limit 59.38 GB.

    #TODO
    # 4. train on GPU with more free memory
    #setup_cuda(visible_devices="0,1", multiGPU=False, use_cuda_with_id=-1)