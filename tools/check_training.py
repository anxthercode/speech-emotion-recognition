"""Check local training dependencies, prepared data and a CUDA operation."""
import json


def main():
    try:
        import torch
        import sklearn
        from ser.dataset import PreparedDataset
    except ImportError as error:
        raise SystemExit(f"Missing dependency: {error}. See TRAINING.md for installation commands.")
    data = PreparedDataset(split="speaker", subset="train")
    available = torch.cuda.is_available()
    if available:
        tensor = torch.ones((64, 64), device="cuda")
        assert float((tensor @ tensor).mean().item()) == 64
    print(json.dumps({"torch": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
                      "cuda_available": available,
                      "gpu": torch.cuda.get_device_name(0) if available else None,
                      "sklearn": sklearn.__version__, "prepared_data": str(data.root),
                      "speaker_train_files": len(data)}, indent=2))


if __name__ == "__main__":
    main()
