import os
import torch
from datasets import load_dataset
from filelock import SoftFileLock
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

HF_PATH = "flwrlabs/femnist"


def _dataset_lock() -> SoftFileLock:
    # FLARE (subprocess-per-site) and FedML (container-per-client) both
    # spawn every client independently with no coordination between them,
    # so on a cold cache they all call load_dataset() for the same dataset
    # at once and race on the HF datasets library's own .incomplete staging
    # dir (one process's cleanup rmtree's it out from under another still
    # mid-build). This lock file lives in the same bind-mounted cache dir
    # every client/container shares, serializing the first (slow) build
    # across processes *and* containers; every later load_dataset() call
    # is a fast local cache read that never contends for the lock.
    lock_dir = os.environ.get("HF_DATASETS_CACHE", "/tmp")
    os.makedirs(lock_dir, exist_ok=True)
    return SoftFileLock(os.path.join(lock_dir, HF_PATH.replace("/", "_") + ".lock"), timeout=600)


def warm_cache():
    with _dataset_lock():
        load_dataset(HF_PATH, split="train")


def load_data(partition_strategy: str, samples_per_client: int, num_clients: int, client_id: int, batch_size: int = 32, alpha: float = 0.5, shards_per_client: int = 2):
    hf_path, image_key, label_key = HF_PATH, "image", "character"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    with _dataset_lock():
        dataset = load_dataset(hf_path, split="train")
    if label_key not in dataset.features:
        label_key = "label"

    labels = np.array(dataset[label_key])
    total_samples = len(labels)

    np.random.seed(42) # Ensure reproducibility across clients
    
    if partition_strategy == "iid":
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        client_indices = splits[client_id]
        
    elif partition_strategy == "shard":
        sorted_indices = np.argsort(labels)
        total_shards = num_clients * shards_per_client
        shard_size = total_samples // total_shards
        shards = [sorted_indices[i*shard_size : (i+1)*shard_size] for i in range(total_shards)]
        np.random.shuffle(shards)
        client_indices = np.concatenate(shards[client_id * shards_per_client : (client_id + 1) * shards_per_client])
        
    elif partition_strategy == "dirichlet":
        num_classes = len(np.unique(labels))
        client_indices_list = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_k = np.where(labels == c)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, splits)
            for i in range(num_clients):
                client_indices_list[i].extend(idx_k_split[i])
        client_indices = np.array(client_indices_list[client_id])
    else:
        raise ValueError("Invalid partition strategy")

    np.random.shuffle(client_indices)
    client_indices = client_indices[:samples_per_client].tolist()

    dataset = dataset.select(client_indices)

    def apply_transforms(batch):
        batch["pixel_values"] = [transform(img.convert("L")) for img in batch[image_key]]
        batch["labels"] = batch[label_key]
        return batch

    dataset = dataset.with_transform(apply_transforms)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    def collate_fn(batch):
        xs = torch.stack([x["pixel_values"] for x in batch])
        ys = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return xs, ys 

    train_loader = DataLoader(split["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(split["test"], batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader