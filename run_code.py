import subprocess
from multiprocessing import Pool

from tqdm import tqdm


def run_command(cmd):
    try:
        result = subprocess.run(
            cmd.split(), check=True, capture_output=False, text=True
        )
        return {"status": "success", "cmd": cmd, "returncode": result.returncode}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "cmd": cmd, "returncode": e.returncode}


def main():
    n_parallel = 4
    models = ["densenet", "resnet", "mobilenet", "efficientnet"]
    layers = [5]
    batch_sizes = [32]
    # datasets = [
    #     "chest_xray",
    #     "CXR8",
    # ]
    # cross_datasets = [
    #     "CXR8",
    #     "chest_xray",
    # ]
    datasets = ["CXR8"]
    cross_datasets = ["chest_xray"]
    losses = [
        "cross_entropy",
        "focal_loss",
    ]
    samplers = [
        "weighted",
        "balanced",
    ]

    configs = []

    for layer in layers:
        for batch_size in batch_sizes:
            for loss in losses:
                for sampler in samplers:
                    for dataset in datasets:
                        for cross in cross_datasets:
                            if cross == dataset:
                                continue
                            for model in models:
                                configs.append(
                                    f"python3 main.py --model {model} --layers {layer} --batch-size {batch_size} --dataset {dataset} --cross {cross} --loss {loss} --sampler {sampler}"
                                )

    print(
        f"Total de configurações: {len(configs)} | Paralelas: {n_parallel} instâncias"
    )

    with Pool(processes=n_parallel) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_command, configs),
                total=len(configs),
                desc="Progresso",
            )
        )

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'=' * 50}")
    print(f"Sucesso: {success_count}")
    print(f"Erros: {error_count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
