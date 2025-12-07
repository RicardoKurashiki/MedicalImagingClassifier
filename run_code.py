import subprocess
from multiprocessing import Pool

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(prog="Medical Imaging Analysis Classifier")

parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot Results",
)

args = parser.parse_args()


def run_command(cmd):
    try:
        result = subprocess.run(
            cmd.split(), check=True, capture_output=False, text=True
        )
        return {"status": "success", "cmd": cmd, "returncode": result.returncode}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "cmd": cmd, "returncode": e.returncode}


def main():
    n_parallel = 6
    models = ["resnet", "densenet"]
    layers = [None, 2, 5]
    batch_sizes = [32]
    datasets = [
        "chest_xray",
        "CXR8",
    ]
    cross_datasets = [
        "CXR8",
        "chest_xray",
    ]
    epochs = 500

    configs = []

    for dataset in datasets:
        for cross in cross_datasets:
            for layer in layers:
                for batch_size in batch_sizes:
                    if cross == dataset:
                        continue
                    for model in models:
                        config = "python3 main.py"
                        if model is not None:
                            config += f" --model {model}"
                        if layer is not None:
                            config += f" --layers {layer}"
                        if batch_size is not None:
                            config += f" --batch-size {batch_size}"
                        if epochs is not None:
                            config += f" --epochs {epochs}"
                        config += f" --dataset {dataset} --cross {cross}"
                        if args.plot:
                            config += " --plot-results"
                        configs.append(config)

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
