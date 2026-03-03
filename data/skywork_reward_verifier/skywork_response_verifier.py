import os
from pathlib import Path
from data.commons import create_standard_task_toml

'''
Templates and logic for the Skywork Reward Verifier (Response File Mode).
This version requires the agent to write its final answer to 'response.txt'.
Instruction is parsed from the trajectory log.
'''

VERIFIER_TEMPLATE = '''
import torch
import os
import sys
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_instruction_from_trajectory(trajectory_path: Path):
    """Extracts the core instruction from the first step of the trajectory."""
    with open(trajectory_path, "r") as f:
        data = json.load(f)
    
    steps = data.get("steps", [])
    if not steps:
        return ""
        
    for i, step in enumerate(steps):
        if step.get("source") == "user" and i == 0:
            message = step.get("message", "")
            
            # Extract content between "Task Description:" and "Current terminal state:"
            if "Task Description:" in message:
                content = message.split("Task Description:", 1)[1]
            else:
                content = message
                
            if "Current terminal state:" in content:
                content = content.split("Current terminal state:", 1)[0]
                
            return content.strip()
    return ""

def run_verifier():
    print("--- Starting Skywork Verification (Response File Mode) ---")
    
    # 1. Load context
    traj_path = Path("/logs/agent/trajectory.json")
    response_path = Path("response.txt")
    reward_file = Path("/logs/verifier/reward.txt")
    
    if not traj_path.exists():
        print(f"Error: {traj_path} not found.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    if not response_path.exists():
        print("Error: response.txt not found. Agent failed to provide the required deliverable.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    # 2. Extract Instruction and Answer
    user_prompt = get_instruction_from_trajectory(traj_path)
    agent_answer = response_path.read_text().strip()

    if not user_prompt:
        print("Error: Could not parse instruction from trajectory.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    if not agent_answer:
        print("Error: response.txt is empty.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    # 3. Format for Skywork (Two-Turn)
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": agent_answer}
    ]
    
    print(f"DEBUG: Messages sent to model: {json.dumps(messages, indent=2)}")

    # 4. Load Model and Tokenizer
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

    # 5. Process and Score
    messages_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    
    if tokenizer.bos_token is not None and messages_formatted.startswith(tokenizer.bos_token):
        messages_formatted = messages_formatted[len(tokenizer.bos_token):]
        
    messages_tokenized = tokenizer(
        messages_formatted, 
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    with torch.no_grad():
        score = rm(**messages_tokenized).logits[0][0].item()

    # 6. Clip Score [0, 100]
    final_reward = score
    if final_reward < 0:
        print(f"Clipping reward: original score {score:.6f} is less than 0")
        final_reward = 0
    elif final_reward > 100:
        print(f"Clipping reward: original score {score:.6f} is greater than 100")
        final_reward = 100

    # 7. Output Reward
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
# Verifier Entrypoint (Response File Mode)

# 1. System-level setup
apt-get update
apt-get install -y curl jq
apt-get install -y python3-pip python3-full

# 2. Install ML dependencies
pip install --break-system-packages torch transformers accelerate

# 3. Run the Skywork Judge script
python3 -u /tests/test_state.py
'''

RESOURCES_TEMPLATE = '''
    [environment]
    cpus = 8
    memory_mb = 16384
    storage_mb = 10240
'''

def inject_skywork_response_verifier(dataset_dir: str):
    """Adds the Skywork Response verifier files and requirements to tasks."""
    tasks_root = Path(dataset_dir)
    print(f"Injecting Skywork Response verifier into tasks at: {tasks_root}")
    
    base_toml = create_standard_task_toml()
    updated_toml = base_toml.replace("timeout_sec = 720.0", "timeout_sec = 1200.0")
    skywork_task_toml = updated_toml.strip() + "\n" + RESOURCES_TEMPLATE + "\n"

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir(): continue
            
        # 1. Update task.toml
        with open(task_dir / "task.toml", "w") as f:
            f.write(skywork_task_toml)

        # 2. Setup tests directory
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        with open(tests_dir / "test_state.py", "w") as f:
            f.write(VERIFIER_TEMPLATE)
            
        test_sh_path = tests_dir / "test.sh"
        with open(test_sh_path, "w") as f:
            f.write(TEST_SH_TEMPLATE)
        os.chmod(test_sh_path, 0o755)

        # 3. Update instruction.md with the deliverable requirement
        instr_path = task_dir / "instruction.md"
        if instr_path.exists():
            instruction = instr_path.read_text()
            
            deliverable_requirement = (
                "After you have completed your analysis and formulated your answer, "
                "you MUST write your final, comprehensive response into a file named "
                "'response.txt' in the current directory."
            )

            if "response.txt" not in instruction:
                instr_path.write_text(instruction + deliverable_requirement)
        
    print("Response-based verifier injection complete.")
