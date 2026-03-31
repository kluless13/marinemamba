"""
Run MarineMamba experiments on Vast.ai GPU.

Usage:
    # Set your API key first
    export VAST_API_KEY="your-key-here"

    # Run everything
    python scripts/run_vast.py

    # Run specific models only
    python scripts/run_vast.py --models A B C D E
    python scripts/run_vast.py --models F  # Evo 2 only

This will:
    1. Search for cheapest A100 on Vast.ai
    2. Rent it, SSH in, clone repo
    3. Install packages, fetch BOLD data
    4. Run all models
    5. Pull results back locally
    6. Destroy the instance
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/kluless13/marinemamba.git"

SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== MarineMamba Vast.ai Setup ==="

# Install system deps
apt-get update -qq && apt-get install -qq -y ncbi-blast+ > /dev/null 2>&1 || true

# Upgrade setuptools (vtx needs license-files)
pip install --upgrade setuptools > /dev/null 2>&1

# Install ML packages (order matters)
echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation 2>&1 | tail -1
echo "Installing vtx..."
pip install vtx --no-build-isolation 2>&1 | tail -1
echo "Installing evo2..."
pip install evo2 --no-build-isolation 2>&1 | tail -1
echo "Installing mamba-ssm..."
pip install mamba-ssm --no-build-isolation 2>&1 | tail -1
echo "Installing causal-conv1d..."
pip install causal-conv1d --no-build-isolation 2>&1 | tail -1
echo "Installing other deps..."
pip install pytorch_lightning einops timm torchtext transformers huggingface_hub tqdm scikit-learn > /dev/null 2>&1

# Clone repo
cd /workspace
git clone {repo_url} 2>/dev/null || (cd marinemamba && git pull)
cd marinemamba

# Verify
python -c "
import torch
print(f'GPU: {{torch.cuda.get_device_name(0)}}')
print(f'VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}} GB')
from mamba_ssm.modules.mamba2 import Mamba2
from causal_conv1d import causal_conv1d_fn
from evo2 import Evo2
print('All packages OK!')
"

echo "=== Setup Complete ==="
""".format(repo_url=REPO_URL)

DATA_SCRIPT = """#!/bin/bash
set -e
cd /workspace/marinemamba

echo "=== Fetching BOLD Data ==="
python scripts/01b_process_bold.py 2>/dev/null || python -c "
import requests, csv, sys, os
from pathlib import Path
csv.field_size_limit(sys.maxsize)

RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)
BOLD_TSV = RAW_DIR / 'bold_teleostei.tsv'

if not BOLD_TSV.exists():
    print('Fetching from BOLD v5 API...')
    BASE = 'https://portal.boldsystems.org/api'
    r = requests.get(f'{BASE}/query', params={'query': 'tax:class:Teleostei', 'extent': 'full'}, timeout=60)
    r.raise_for_status()
    query_id = r.json()['query_id']
    r = requests.get(f'{BASE}/documents/{query_id}/download', params={'format': 'tsv'}, timeout=600, stream=True)
    r.raise_for_status()
    with open(BOLD_TSV, 'wb') as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    print(f'Downloaded: {BOLD_TSV}')

print('Processing...')
os.system('python scripts/01b_process_bold.py')
"

echo "=== Running Splits ==="
python scripts/02_clean_and_split.py
cat data/processed/dataset_stats.json
"""

MODEL_COMMANDS = {
    "AB": "python scripts/03_baselines.py",
    "C": "python scripts/04_barcodemamba_models.py --mode transfer --data-dir data/processed --output-dir results",
    "D": "python scripts/04_barcodemamba_models.py --mode scratch --data-dir data/processed --output-dir results",
    "E": "python scripts/04_barcodemamba_models.py --mode adapt --data-dir data/processed --output-dir results",
    "F": "python -c 'import torch,gc; gc.collect(); torch.cuda.empty_cache()' && python scripts/05_evo2_embeddings.py --data-dir data/processed --output-dir results",
}


def run_cmd(cmd, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def search_offers(gpu_name="A100", min_ram=39, max_price=2.0):
    """Search for GPU offers on Vast.ai."""
    print(f"Searching for {gpu_name} (min {min_ram}GB, max ${max_price}/hr)...")

    from vastai import VastAI
    vast = VastAI()

    offers = vast.search_offers(
        query=f"gpu_name={gpu_name} gpu_ram>={min_ram} dph<={max_price} reliability>0.95 inet_down>200 disk_space>=100",
        order="dph",
        limit=5,
    )
    return offers


def create_instance(offer_id, image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"):
    """Create a Vast.ai instance."""
    print(f"Creating instance from offer {offer_id}...")

    from vastai import VastAI
    vast = VastAI()

    result = vast.create_instances(
        ID=offer_id,
        image=image,
        disk=100,
    )
    return result


def wait_for_instance(instance_id, timeout=300):
    """Wait for instance to be ready."""
    from vastai import VastAI
    vast = VastAI()

    print(f"Waiting for instance {instance_id} to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        instances = vast.show_instances()
        for inst in instances:
            if str(inst.get("id")) == str(instance_id):
                status = inst.get("actual_status", "unknown")
                if status == "running":
                    ssh_host = inst.get("ssh_host")
                    ssh_port = inst.get("ssh_port")
                    print(f"  Ready! SSH: ssh -p {ssh_port} root@{ssh_host}")
                    return inst
                print(f"  Status: {status}...")
        time.sleep(10)
    raise TimeoutError("Instance did not start in time")


def ssh_exec(inst, command, timeout=7200):
    """Execute a command on the instance via SSH."""
    ssh_host = inst["ssh_host"]
    ssh_port = inst["ssh_port"]
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} '{command}'"
    print(f"\n>>> {command[:80]}...")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr[-500:]}")
    return result


def pull_results(inst, local_dir="results"):
    """SCP results back to local machine."""
    ssh_host = inst["ssh_host"]
    ssh_port = inst["ssh_port"]
    os.makedirs(local_dir, exist_ok=True)
    scp_cmd = f"scp -o StrictHostKeyChecking=no -P {ssh_port} -r root@{ssh_host}:/workspace/marinemamba/results/* {local_dir}/"
    print(f"Pulling results to {local_dir}/...")
    subprocess.run(scp_cmd, shell=True)

    scp_stats = f"scp -o StrictHostKeyChecking=no -P {ssh_port} root@{ssh_host}:/workspace/marinemamba/data/processed/dataset_stats.json {local_dir}/"
    subprocess.run(scp_stats, shell=True)


def destroy_instance(instance_id):
    """Destroy the instance."""
    from vastai import VastAI
    vast = VastAI()
    print(f"Destroying instance {instance_id}...")
    vast.destroy_instances(id=instance_id)
    print("  Destroyed.")


def main():
    parser = argparse.ArgumentParser(description="Run MarineMamba on Vast.ai")
    parser.add_argument("--models", nargs="+", default=["AB", "C", "D", "E", "F"],
                        help="Which models to run (AB, C, D, E, F)")
    parser.add_argument("--gpu", default="A100", help="GPU type to search for")
    parser.add_argument("--max-price", type=float, default=2.0, help="Max $/hr")
    parser.add_argument("--skip-setup", action="store_true", help="Skip package installation")
    parser.add_argument("--instance-id", type=str, help="Reuse existing instance")
    args = parser.parse_args()

    if not os.environ.get("VAST_API_KEY"):
        print("ERROR: Set VAST_API_KEY environment variable first")
        print("  Get your key at: https://cloud.vast.ai/api/")
        sys.exit(1)

    inst = None
    instance_id = args.instance_id

    try:
        if not instance_id:
            # Search and create
            offers = search_offers(gpu_name=args.gpu, max_price=args.max_price)
            if not offers:
                print("No offers found. Try increasing --max-price or changing --gpu")
                sys.exit(1)

            print(f"\nFound {len(offers)} offers. Using cheapest.")
            offer = offers[0]
            result = create_instance(offer["id"])
            instance_id = result.get("new_contract")
            print(f"Instance ID: {instance_id}")

        # Wait for ready
        inst = wait_for_instance(instance_id)

        # Setup
        if not args.skip_setup:
            print("\n" + "=" * 60)
            print("SETTING UP ENVIRONMENT")
            print("=" * 60)
            ssh_exec(inst, SETUP_SCRIPT, timeout=3600)

        # Fetch data
        print("\n" + "=" * 60)
        print("FETCHING & PROCESSING DATA")
        print("=" * 60)
        ssh_exec(inst, DATA_SCRIPT, timeout=600)

        # Run models
        for model_key in args.models:
            if model_key not in MODEL_COMMANDS:
                print(f"Unknown model: {model_key}")
                continue

            print(f"\n{'=' * 60}")
            print(f"RUNNING MODEL {model_key}")
            print("=" * 60)
            cmd = f"cd /workspace/marinemamba && {MODEL_COMMANDS[model_key]}"
            ssh_exec(inst, cmd, timeout=7200)

        # Pull results
        print(f"\n{'=' * 60}")
        print("PULLING RESULTS")
        print("=" * 60)
        pull_results(inst)

        # Show results
        print(f"\n{'=' * 60}")
        print("DONE! Results saved to ./results/")
        print("=" * 60)
        for f in sorted(os.listdir("results")):
            if f.endswith(".json"):
                print(f"  {f}")

    finally:
        if inst and not args.instance_id:
            confirm = input("\nDestroy instance? (y/n): ").strip().lower()
            if confirm == "y":
                destroy_instance(instance_id)
            else:
                print(f"Instance still running. Destroy manually: vastai destroy instance {instance_id}")


if __name__ == "__main__":
    main()
