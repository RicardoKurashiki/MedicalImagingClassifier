import os
import time
import psutil
import torch
import torch.nn as nn
from tqdm import tqdm
from tempfile import TemporaryDirectory

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def get_memory_usage():
    metrics = {}

    # CPU
    process = psutil.Process(os.getpid())
    metrics["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    # GPU
    if torch.cuda.is_available():
        metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        metrics["gpu_memory_max_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
        )
        metrics["gpu_utilization"] = (
            torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else None
        )
    else:
        metrics["gpu_memory_allocated_mb"] = None
        metrics["gpu_memory_reserved_mb"] = None
        metrics["gpu_memory_max_allocated_mb"] = None
        metrics["gpu_utilization"] = None

    return metrics


def get_model_size(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


def train_model(
    model,
    pretrained_model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=100,
    early_stopping_patience=10,
    verbose=False,
):
    since = time.time()

    metrics = {
        "total_time_seconds": 0,
        "epoch_times": [],
        "batch_times": {"train": [], "val": []},
        "throughput": {"train": [], "val": []},  # samples/segundo
        "memory_usage": [],
        "model_info": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "model_size_mb": get_model_size(model),
        },
    }

    # Memória inicial
    initial_memory = get_memory_usage()
    metrics["initial_memory"] = initial_memory

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(
            tempdir, f"{pretrained_model}_best_model_params.pt"
        )
        if verbose:
            print(f"Melhores pesos salvos em {best_model_params_path}")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        early_stop = False

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            if early_stop:
                if verbose:
                    print(f"Early stopping acionado na época {epoch}")
                break
            epoch_start = time.time()
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
            if verbose:
                print("-" * 10)

            for phase in ["train", "val"]:
                phase_start = time.time()
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples_processed = 0
                batch_times = []
                if verbose:
                    pbar = tqdm(
                        dataloaders[phase],
                        desc=f"{phase.capitalize():5s}",
                        unit="batch",
                        leave=False,
                    )
                else:
                    pbar = dataloaders[phase]

                for inputs, labels in pbar:
                    batch_start = time.time()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    total_samples_processed += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(
                        0
                    )
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                phase_time = time.time() - phase_start
                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.double() / total_samples_processed

                # Salvar métricas computacionais da fase
                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )

                metrics["batch_times"][phase].append(
                    {
                        "epoch": epoch,
                        "avg_batch_time_seconds": avg_batch_time,
                        "total_batches": len(batch_times),
                        "total_time_seconds": phase_time,
                    }
                )

                metrics["throughput"][phase].append(
                    {
                        "epoch": epoch,
                        "total_samples": total_samples_processed,
                        "total_time_seconds": phase_time,
                    }
                )

                if verbose:
                    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                if verbose:
                    print(f"{phase} Time: {phase_time:.2f}s")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())

                if verbose:
                    print(
                        f"{phase.capitalize():5s} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
                    )

                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        if verbose:
                            print(f"Melhor acc [{phase}]: {best_acc:.4f}")

                    if epoch_loss < best_val_loss:
                        past_loss = best_val_loss
                        best_val_loss = epoch_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_params_path)
                        if verbose:
                            print(
                                f"Loss val melhorou de {past_loss:.4f} para {best_val_loss:.4f}"
                            )
                    else:
                        patience_counter += 1
                        if verbose:
                            print(
                                f"Loss val não melhorou de {best_val_loss:.4f} - Paciência: {patience_counter}"
                            )
                        if patience_counter >= early_stopping_patience:
                            early_stop = True
                            if verbose:
                                print("Early stopping acionado")
                            break

            # Verificar early stopping após o loop de fases
            if early_stop:
                break

            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append({"epoch": epoch, "time_seconds": epoch_time})

            # Coletar memória após cada época
            epoch_memory = get_memory_usage()
            metrics["memory_usage"].append({"epoch": epoch, **epoch_memory})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if verbose:
                print()

        time_elapsed = time.time() - since
        metrics["total_time_seconds"] = time_elapsed

        final_memory = get_memory_usage()
        metrics["final_memory"] = final_memory

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if verbose:
            print(
                f"Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        if verbose:
            print(f"Melhor acc [val]: {best_acc:4f}")

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, history, metrics
