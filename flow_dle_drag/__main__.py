from .config import parse_args
from .benchmark import benchmark_mode

def main():
    """Entry point for benchmark mode."""
    args = parse_args()
    
    if args.mode == "benchmark":
        benchmark_mode(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()