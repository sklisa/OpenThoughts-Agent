#!/usr/bin/env python3
"""
Generate Stack Overflow dataset with Skywork Reward verifier.
This version extracts from an existing HF tasks dataset instead of raw XML.

Sample usage:
    # Use standard multi-turn trajectory verifier
    python3 data/skywork_reward_verifier/generate_overflow.py --verifier_type standard

    # Use response.txt based verifier
    python3 data/skywork_reward_verifier/generate_overflow.py --verifier_type response

    # Local test (No Upload)
    python3 data/skywork_reward_verifier/generate_overflow.py --verifier_type response --skip_upload
"""

import tempfile
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from parent package
from data.commons import (
    upload_tasks_to_hf, 
    download_hf_dataset
)
from scripts.harbor import tasks_parquet_converter as tpc

# Import both Skywork verifiers
from data.skywork_reward_verifier.skywork_verifier import inject_skywork_verifier
from data.skywork_reward_verifier.skywork_response_verifier import inject_skywork_response_verifier

def main() -> None:
    """Main function - processes StackOverflow tasks with chosen Skywork Reward verifier"""
    parser = argparse.ArgumentParser(description="Generate Stack Overflow dataset with Skywork Reward")
    parser.add_argument(
        "--verifier_type", 
        choices=["standard", "response"], 
        default="standard",
        help="Which Skywork verifier to use: 'standard' (trajectory) or 'response' (response.txt)"
    )
    parser.add_argument("--skip_upload", action="store_true", help="Skip upload to Hugging Face")
    args = parser.parse_args()
    
    source_repo = "mlfoundations-dev/stackexchange-overflow-sandboxes"
    
    print(f"Step 1: Downloading source tasks from {source_repo}...")
    snapshot_dir = Path(download_hf_dataset(source_repo))
    
    # 2. Extract tasks from parquet
    print("Step 2: Extracting tasks from all parquet files...")
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {snapshot_dir}")
    
    output_dir = Path(tempfile.mkdtemp(prefix="overflow_skywork_"))
    print(f"Extracting to: {output_dir}")
    
    for i, parquet_file in enumerate(parquet_files):
        current_on_exist = "overwrite" if i == 0 else "skip"
        tpc.from_parquet(
            parquet_path=str(parquet_file),
            base=str(output_dir),
            on_exist=current_on_exist
        )

    # 3. Verifier Injection
    if args.verifier_type == "response":
        print("Step 3: Injecting local Skywork Response verifier...")
        inject_skywork_response_verifier(str(output_dir))
        suffix = "-skywork-response"
    else:
        print("Step 3: Injecting local Skywork Standard verifier...")
        inject_skywork_verifier(str(output_dir))
        suffix = "-skywork"

    target_repo = f"DCAgent/stackexchange-overflow-sandboxes{suffix}"

    # 4. Upload Tasks
    if not args.skip_upload:
        print(f"Step 4: Uploading verified tasks to {target_repo}...")
        upload_tasks_to_hf(str(output_dir), target_repo)
        print(f"Success! Tasks uploaded to: https://huggingface.co/datasets/{target_repo}")
    else:
        print(f"Upload skipped. Local tasks are available in: {output_dir}")
    
    print("Generation Complete!")

if __name__ == "__main__":
    main()
