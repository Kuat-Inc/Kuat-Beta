"""
Kuat CLI - Dataset info and benchmarking

Usage:
    kuat info <archive>
    kuat benchmark <archive>

Note: Dataset conversion is done using the `quat-tree` binary:
    quat-tree vq-create ./imagenet/train -o train.qvq --format imagenet
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kuat",
        description="Ultra-fast ML dataset loading",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {get_version()}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show archive information")
    info_parser.add_argument("archive", help="Archive file path (.qvq)")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark dataloader speed")
    bench_parser.add_argument("archive", help="Archive file path (.qvq)")
    bench_parser.add_argument("--batch-size", "-b", type=int, default=64,
                              help="Batch size (default: 64)")
    bench_parser.add_argument("--epochs", "-e", type=int, default=1,
                              help="Number of epochs (default: 1)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\nNote: To convert datasets, use the quat-tree binary:")
        print("  quat-tree vq-create ./data -o dataset.qvq --format imagenet")
        sys.exit(0)
    
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def convert():
    """Direct convert entry point - shows instructions."""
    print("Dataset conversion is done using the quat-tree binary:")
    print()
    print("  # Convert ImageNet-style folder")
    print("  quat-tree vq-create ./imagenet/train -o train.qvq --format imagenet")
    print()
    print("  # Convert CIFAR-10")
    print("  quat-tree vq-create ./cifar-10-batches-py -o cifar.qvq --format cifar")
    print()
    print("  # See all options")
    print("  quat-tree vq-create --help")
    sys.exit(0)


def cmd_info(args):
    """Handle info command."""
    from kuat import KuatArchive
    
    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"Error: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        archive = KuatArchive(str(archive_path))
        info = archive.info()
        
        print(f"Archive: {info['path']}")
        print(f"  Samples: {info['samples']:,}")
        print(f"  Dimensions: {info['width']}x{info['height']}")
        print(f"  Bits: {info['bits']}")
        if info.get('classes'):
            print(f"  Classes: {len(info['classes'])}")
            if len(info['classes']) <= 20:
                print(f"    {', '.join(info['classes'][:10])}")
                if len(info['classes']) > 10:
                    print(f"    ... and {len(info['classes']) - 10} more")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    return 0


def cmd_benchmark(args):
    """Handle benchmark command."""
    from kuat import KuatDataset
    
    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"Error: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        dataset = KuatDataset(str(archive_path), batch_size=args.batch_size)
        
        print(f"Archive: {archive_path}")
        print(f"Samples: {len(dataset):,}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print()
        
        total_samples = 0
        total_time = 0.0
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            batch_count = 0
            sample_count = 0
            
            for batch in dataset.epoch(epoch):
                batch_count += 1
                sample_count += batch["images"].shape[0]
            
            epoch_time = time.time() - epoch_start
            samples_per_sec = sample_count / epoch_time
            
            print(f"Epoch {epoch}: {batch_count} batches, {sample_count:,} samples, "
                  f"{epoch_time:.2f}s ({samples_per_sec:,.0f} samples/sec)")
            
            total_samples += sample_count
            total_time += epoch_time
        
        print()
        print(f"Total: {total_samples:,} samples in {total_time:.2f}s")
        print(f"Average: {total_samples/total_time:,.0f} samples/sec")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    return 0


def get_version():
    """Get package version."""
    try:
        from kuat import __version__
        return __version__
    except ImportError:
        return "0.1.0-beta.1"


if __name__ == "__main__":
    sys.exit(main() or 0)
