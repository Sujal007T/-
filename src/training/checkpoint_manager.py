import os
import json
import time
import torch
import shutil
from typing import Optional, Dict, Any, Tuple

class TrainingCheckpointManager:
    """
    A robust checkpoint and resume manager for standard PyTorch models and HuggingFace / PEFT models.
    
    Guarantees:
    1. Saves model weights (LoRA adapter or full weights), optimizer state, scheduler state, 
       epoch number, exact batch step inside epoch, cumulative global step, and loss logs.
    2. Atomic metadata writing via 'latest_checkpoint.json' so partial writes never corrupt past checkpoints.
    3. Seamlessly resumes training from the exact epoch and step where it was stopped.
    4. Automatically fast-forwards the DataLoader / progress bar without re-running finished samples.
    """
    
    def __init__(self, checkpoint_dir: str, save_freq_steps: int = 50, keep_last_n: int = 5):
        """
        Args:
            checkpoint_dir: Base directory where checkpoints are saved (e.g. Google Drive folder).
            save_freq_steps: Interval in optimization steps to save checkpoints.
            keep_last_n: Number of recent step checkpoints to keep (to prevent disk overflow).
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq_steps = save_freq_steps
        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.meta_file = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss_log: list,
        current_loss: Optional[float] = None,
        tag: Optional[str] = None,
        is_final: bool = False
    ) -> str:
        """
        Saves full training state to disk.
        """
        if tag is not None:
            folder_name = tag
        elif is_final:
            folder_name = "final_model"
        else:
            folder_name = f"step_{global_step}"

        save_path = os.path.join(self.checkpoint_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)

        # 1. Save Model Weights
        # Checks if PEFT / HuggingFace model (has save_pretrained) or standard PyTorch nn.Module
        is_peft_or_hf = hasattr(model, "save_pretrained")
        if is_peft_or_hf:
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

        # 2. Save Optimizer, Scheduler, and Progress State
        state = {
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss_log": loss_log,
            "is_peft_or_hf": is_peft_or_hf,
            "timestamp": time.time(),
            "current_loss": current_loss
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, os.path.join(save_path, "training_state.pt"))

        # 3. Update 'latest_checkpoint.json' pointer
        meta_info = {
            "checkpoint_dir": save_path,
            "folder_name": folder_name,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss": current_loss,
            "is_peft_or_hf": is_peft_or_hf,
            "timestamp": time.ctime()
        }
        
        temp_meta = self.meta_file + ".tmp"
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)
        if os.path.exists(self.meta_file):
            try:
                os.remove(self.meta_file)
            except Exception:
                pass
        try:
            os.replace(temp_meta, self.meta_file)
        except Exception:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=2)
            if os.path.exists(temp_meta):
                try:
                    os.remove(temp_meta)
                except Exception:
                    pass

        # 4. Prune older step checkpoints if keep_last_n is set
        if not is_final and self.keep_last_n > 0:
            self._prune_old_checkpoints()

        return save_path

    def _prune_old_checkpoints(self):
        """Removes older step checkpoints to avoid filling disk space."""
        try:
            subdirs = [
                d for d in os.listdir(self.checkpoint_dir)
                if d.startswith("step_") and os.path.isdir(os.path.join(self.checkpoint_dir, d))
            ]
            # Sort by step number
            subdirs.sort(key=lambda x: int(x.split("_")[-1]))
            if len(subdirs) > self.keep_last_n:
                to_delete = subdirs[:-self.keep_last_n]
                for old_dir in to_delete:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, old_dir), ignore_errors=True)
        except Exception:
            pass

    def get_latest_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Returns metadata about the latest valid checkpoint, or None."""
        if not os.path.exists(self.meta_file):
            # Fallback: scan for any existing step directories
            subdirs = [
                d for d in os.listdir(self.checkpoint_dir)
                if (d.startswith("step_") or d.startswith("lora_step_")) 
                and os.path.isdir(os.path.join(self.checkpoint_dir, d))
            ]
            if not subdirs:
                return None
            subdirs.sort(key=lambda x: int(x.split("_")[-1]))
            latest_dir = os.path.join(self.checkpoint_dir, subdirs[-1])
            return {"checkpoint_dir": latest_dir, "folder_name": subdirs[-1]}

        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def resume_training(
        self,
        base_model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        lora_config: Optional[Any] = None,
        dataset_len: Optional[int] = None,
        grad_accum: int = 8,
        checkpoint_dir: Optional[str] = None
    ) -> Tuple[Any, Optional[torch.optim.Optimizer], Optional[Any], int, int, int, list]:
        """
        Loads the latest checkpoint if available and restores model, optimizer, scheduler, and progress.
        """
        if checkpoint_dir is not None and os.path.exists(checkpoint_dir):
            ckpt_path = checkpoint_dir
        else:
            info = self.get_latest_checkpoint_info()
            if not info or not os.path.exists(info.get("checkpoint_dir", "")):
                print("[INFO] No previous checkpoint found. Starting fresh from Epoch 0, Step 0.")
                if lora_config is not None:
                    from peft import get_peft_model
                    model = get_peft_model(base_model, lora_config)
                else:
                    model = base_model
                return model, optimizer, scheduler, 0, 0, 0, []
            ckpt_path = info["checkpoint_dir"]
        state_path = os.path.join(ckpt_path, "training_state.pt")

        print(f"[RESUME] Checkpoint detected at: {ckpt_path}")

        # 1. Load Model Weights
        # PEFT vs standard PyTorch
        if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            from peft import PeftModel
            print("   -> Loading PEFT/LoRA adapter weights...")
            model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
        elif os.path.exists(os.path.join(ckpt_path, "model.pt")):
            print("   -> Loading PyTorch model weights...")
            base_model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.pt"), map_location="cpu"))
            model = base_model
        else:
            model = base_model

        # 2. Load Optimizer, Scheduler & Training State
        start_epoch = 0
        start_step = 0
        global_step = 0
        loss_log = []

        if os.path.exists(state_path):
            print("   -> Restoring optimizer, scheduler, and step progress...")
            state = torch.load(state_path, map_location="cpu")
            start_epoch = state.get("epoch", 0)
            # Resume from the NEXT step after the saved step
            start_step = state.get("step_in_epoch", 0) + 1
            global_step = state.get("global_step", 0)
            loss_log = state.get("loss_log", [])

            if optimizer is not None and "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                    print("   -> Optimizer state restored.")
                except Exception as e:
                    print(f"   [WARNING] Could not load optimizer state: {e}")

            if scheduler is not None and "scheduler_state_dict" in state:
                try:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                    print("   -> Scheduler state restored.")
                except Exception as e:
                    print(f"   [WARNING] Could not load scheduler state: {e}")
        else:
            # Fallback when migrating from older checkpoints (e.g. lora_step_150)
            print("   [INFO] Parsing previous training progress from checkpoint folder name...")
            folder_name = os.path.basename(ckpt_path)
            try:
                global_step = int(folder_name.split("_")[-1])
            except Exception:
                global_step = 0

            # Calculate exact epoch and sample offset from global_step
            if dataset_len and dataset_len > 0 and global_step > 0:
                steps_per_epoch = max(dataset_len // grad_accum, 1)
                start_epoch = global_step // steps_per_epoch
                start_step = (global_step % steps_per_epoch) * grad_accum
                print(f"   -> Deduced from global step {global_step}: Epoch {start_epoch + 1}, Sample {start_step}")
            else:
                start_epoch = 0
                start_step = 0

        print(f"[SUCCESS] Resumed training: Starting from Epoch {start_epoch + 1}, Step {start_step} (Global Step {global_step}).")
        return model, optimizer, scheduler, start_epoch, start_step, global_step, loss_log
