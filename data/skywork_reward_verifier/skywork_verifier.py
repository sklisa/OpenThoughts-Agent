import os
from pathlib import Path
from data.commons import create_standard_task_toml

'''
Templates and logic for the Skywork Reward Verifier.
'''

VERIFIER_TEMPLATE = '''
import torch
import os
import sys
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def parse_trajectory_authentic_multiturn(trajectory_path: Path):
    """
    Parses ATIF trajectory JSON into a multi-turn message list.
    Preserves authentic content while mapping sources to roles.
    Extracts the core instruction for the first turn.
    Ignores the final observation.
    """
    with open(trajectory_path, "r") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        return []

    messages = []
    num_steps = len(steps)

    for i, step in enumerate(steps):
        source = step.get("source")
        message = step.get("message", "")

        if source == "user":
            if i == 0:
                # 1. Parse the core task instruction from the first user message
                if "Task Description:" in message:
                    content = message.split("Task Description:", 1)[1]
                else:
                    content = message
                    
                if "Current terminal state:" in content:
                    content = content.split("Current terminal state:", 1)[0]
                    
                messages.append({"role": "user", "content": content.strip()})
            else:
                # Subsequent user interventions
                messages.append({"role": "user", "content": message})

        elif source == "agent":
            # Build the Assistant's turn (Only Thought, skipping Tool Calls)
            messages.append({"role": "assistant", "content": message})
            
            # 3. Add Observation as next 'user' turn, unless it is the final step
            if i < num_steps - 1:
                observation = step.get("observation", {}).get("results", [])
                obs_content = "\\n".join([res.get("content", "") for res in observation if res.get("content")])
                
                if obs_content:
                    messages.append({"role": "user", "content": f"Observation:\\n{obs_content}"})
                
    print(f"DEBUG: Messages sent to model: {json.dumps(messages, indent=2)}")
    return messages

def run_verifier():
    print("--- Starting Skywork Verification (Raw Score Mode) ---")
    
    # 1. Load context from Harbor logs
    traj_path = Path("/logs/agent/trajectory.json")
    reward_file = Path("/logs/verifier/reward.txt")
    
    if not traj_path.exists():
        print(f"Error: {traj_path} not found.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0")
        return

    messages = parse_trajectory_authentic_multiturn(traj_path)
    
    if not messages:
        print("Error: No valid steps found in trajectory.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0")
        return

    # Ensure the last message is from the assistant
    if messages[-1]["role"] != "assistant":
        messages.pop()

    print(f"Parsed {len(messages)} turns from trajectory.")

    # 2. Load Model and Tokenizer
    model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    print(f"Loading Reward Model: {model_name}")
    
    print(f"Initializing Skywork model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        num_labels=1,
    )
    rm.eval()

    # 3. Process and Score
    print("Formatting multi-turn messages...")
    # Important: no system prompt as per Skywork docs
    messages_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Remove the potential duplicate bos token
    if tokenizer.bos_token is not None and messages_formatted.startswith(tokenizer.bos_token):
        messages_formatted = messages_formatted[len(tokenizer.bos_token):]
        
    print("Tokenizing input sequence...")
    messages_tokenized = tokenizer(
        messages_formatted, 
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    print("Executing model forward pass...")
    with torch.no_grad():
        # Everything except the last assistant message is context
        # The last assistant message is what gets scored
        score = rm(**messages_tokenized).logits[0][0].item()

    # 4. Clip Score [0, 100]
    final_reward = score
    if final_reward < 0:
        print(f"Clipping reward: original score {score:.6f} is less than 0")
        final_reward = 0
    elif final_reward > 100:
        print(f"Clipping reward: original score {score:.6f} is greater than 100")
        final_reward = 100

    # 5. Output Reward
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(f"{final_reward:.6f}")
    
    print(f"Verification Complete.")
    print(f"Final Reward: {final_reward:.6f}")

if __name__ == "__main__":
    try:
        run_verifier()
    except Exception as e:
        print(f"Verifier Crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

TEST_SH_TEMPLATE = '''#!/bin/bash
# 1. System-level setup
apt-get update
apt-get install -y curl jq
apt-get install -y python3-pip python3-full

# 2. Install ML dependencies using system-package bypass
pip install --break-system-packages torch transformers accelerate

# 3. Run the Skywork Judge script
python3 -u /tests/test_state.py
'''

# Template for Skywork hardware requirements
RESOURCES_TEMPLATE = '''
    [environment]
    cpus = 8
    memory_mb = 16384
    storage_mb = 10240
'''

def inject_skywork_verifier(dataset_dir: str):
    """Adds the Skywork verifier files to each task directory."""
    tasks_root = Path(dataset_dir)
    print(f"Injecting Skywork verifier into tasks at: {tasks_root}")
    
    # Get the baseline TOML and update the timeout
    base_toml = create_standard_task_toml()
    # Increase verifier timeout from 720 to 1200 seconds
    updated_toml = base_toml.replace("timeout_sec = 720.0", "timeout_sec = 1200.0")
    
    # Append Skywork-specific hardware requirements
    skywork_task_toml = updated_toml.strip() + "\n" + RESOURCES_TEMPLATE + "\n"

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir(): continue
            
        # Write the customized task.toml
        with open(task_dir / "task.toml", "w") as f:
            f.write(skywork_task_toml)

        # Setup tests directory
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Write the python logic to test_state.py
        with open(tests_dir / "test_state.py", "w") as f:
            f.write(VERIFIER_TEMPLATE)
            
        # Write the bash entrypoint
        test_sh_path = tests_dir / "test.sh"
        with open(test_sh_path, "w") as f:
            f.write(TEST_SH_TEMPLATE)
        os.chmod(test_sh_path, 0o755)
        
    print("Verifier injection complete.")
