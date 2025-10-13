import torch
import torch.multiprocessing as mp
import json
import os
import copy
from tqdm import tqdm
from loguru import logger as logu
import argparse

from config import DATASET_REGISTRY, MODEL_REGISTRY


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'eval_results')


def worker_main(model_class, dataset_class, model_path: str, gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """
    每个子进程执行的主函数。它负责初始化模型，然后从任务队列中获取数据项进行处理。
    """
    try:
        device = f'cuda:{gpu_id}'
        # 在子进程中创建模型实例
        logu.info(f"Worker on GPU {gpu_id} starting initialization...")
        worker_instance = model_class(model_path, device)
        logu.info(f"Worker on GPU {gpu_id} initialized successfully.")
        # 获取数据处理函数
        process_func = dataset_class.process_item
        
        while True:
            item = task_queue.get()
            if item is None:  # 收到 None 信号，表示所有任务已分发完毕
                break
            try:
                # 调用数据处理函数
                result = process_func(worker_instance, item)
                result_queue.put(result)
            except Exception as e:
                # 即使处理失败，也记录错误信息并继续
                error_info = {"error": f"Processing failed: {e}", "uniq_id": item.get('uniq_id', 'unknown')}
                result_queue.put(error_info)

    except Exception as e:
        logu.error(f"Worker on GPU {gpu_id} encountered a fatal error: {e}")
    finally:
        logu.info(f"Worker on GPU {gpu_id} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlideASR-Bench Evaluation Tool")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_REGISTRY.keys())
    parser.add_argument("--models-per-gpu", type=int, default=3)
    args = parser.parse_args()

    output_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    args.output_file = f"{output_dir}/{args.model}.jsonl"
    logu.info(args)

    model_class, model_path = MODEL_REGISTRY[args.model]
    dataset_class = DATASET_REGISTRY[args.dataset]
    
    num_gpus = torch.cuda.device_count()
    total_workers = num_gpus * args.models_per_gpu
    logu.info(f"Detected {num_gpus} GPUs. {args. models_per_gpu} model instances will be run on each GPU.")
    logu.info(f"A total of {total_workers} worker processes will be created.")

    # 使用 'spawn' 方法创建子进程，以确保每个子进程都有独立的内存空间
    mp.set_start_method("spawn", force=True)

    # 创建共享队列
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # 数据集初始化，并把所有任务放入任务队列
    dataset = dataset_class(args, model_class)
    for item in dataset:
        task_queue.put(item)
    total_tasks = len(dataset)
    
    # 启动 worker 进程
    worker_processes = []
    for i in range(total_workers):
        gpu_id = i % num_gpus
        p = mp.Process(
            target=worker_main, 
            args=(model_class, dataset_class, model_path, gpu_id, task_queue, result_queue)
        )
        p.start()
        worker_processes.append(p)

    # 收集结果
    processed_count = 0
    with open(args.output_file, 'a') as fw:
        with tqdm(total=total_tasks) as pbar:
            while processed_count < total_tasks:
                result = result_queue.get(timeout=300)
                if result and 'error' not in result:
                    fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                elif result:
                    logu.warning(f"{result.get('uniq_id')}: {result.get('error')}")
                processed_count += 1
                pbar.update(1)
    
    # 发送结束信号给所有 worker
    for _ in range(total_workers):
        task_queue.put(None)
    
    # 等待所有 worker 进程结束
    for p in worker_processes:
        p.join()

    logu.info("All processing has been completed.")
