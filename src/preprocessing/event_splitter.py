import os
import json
import random
import yaml
from collections import defaultdict

def load_jsonl(filepath: str) -> list:
    """Loads a JSONL file. Returns list of dicts."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading JSONL from {filepath}: {e}")
    return data

def event_based_split(data: list, train_ratio: float = 0.8, 
                       event_id_field: str = "event_id",
                       seed: int = 42) -> tuple:
    """
    Groups records by event_id and splits at the EVENT level.
    
    Args:
        data: List of sample dicts
        train_ratio: Fraction of EVENTS (not samples) for training
        event_id_field: Name of the event identifier field
        seed: Random seed for reproducibility
    
    Returns:
        (train_data, val_data) — two lists of sample dicts
    
    After splitting, prints:
    - Total events, train events, val events
    - Total samples in train, total samples in val
    - Samples per disaster type in train vs val (as a table)
    - WARNING if any disaster type has <20 samples in val set
    """
    events = defaultdict(list)
    for record in data:
        event_id = record.get(event_id_field, "UNKNOWN")
        events[event_id].append(record)
        
    event_ids = list(events.keys())
    
    rng = random.Random(seed)
    rng.shuffle(event_ids)
    
    train_end = int(len(event_ids) * train_ratio)
    train_event_ids = event_ids[:train_end]
    val_event_ids = event_ids[train_end:]
    
    train_data = []
    for eid in train_event_ids:
        train_data.extend(events[eid])
        
    val_data = []
    for eid in val_event_ids:
        val_data.extend(events[eid])
        
    print(f"Total events: {len(event_ids)}, Train events: {len(train_event_ids)}, Val events: {len(val_event_ids)}")
    print(f"Total samples in train: {len(train_data)}, Total samples in val: {len(val_data)}")
    
    train_counts = defaultdict(int)
    for r in train_data:
        train_counts[r.get("disaster_type", "UNKNOWN")] += 1
        
    val_counts = defaultdict(int)
    for r in val_data:
        val_counts[r.get("disaster_type", "UNKNOWN")] += 1
        
    all_types = set(train_counts.keys()).union(set(val_counts.keys()))
    
    print("\nSamples per disaster type:")
    print(f"{'Disaster Type':<20} | {'Train':<10} | {'Val':<10}")
    print("-" * 45)
    for d_type in sorted(list(all_types)):
        v_count = val_counts[d_type]
        print(f"{d_type:<20} | {train_counts[d_type]:<10} | {v_count:<10}")
        if v_count < 20:
            print(f"WARNING: Disaster type '{d_type}' has <20 samples in val set ({v_count} samples).")
            
    return train_data, val_data

def verify_no_event_leakage(train_data: list, val_data: list,
                              event_id_field: str = "event_id") -> bool:
    """
    Verifies that no event_id appears in both train and val sets.
    Prints PASS or FAIL with details.
    Returns True if no leakage detected.
    This is a safety check — must be called after every split.
    """
    train_events = set([r.get(event_id_field, "UNKNOWN") for r in train_data])
    val_events = set([r.get(event_id_field, "UNKNOWN") for r in val_data])
    
    leakage = train_events.intersection(val_events)
    
    if len(leakage) > 0:
        print(f"FAIL: Data leakage detected! {len(leakage)} events appear in both sets.")
        print(f"Leaked events: {list(leakage)[:5]}...")
        return False
    else:
        print("PASS: No event leakage detected between train and val sets.")
        return True

def run_split_pipeline(processed_dir: str, output_dir: str,
                        train_ratio: float = 0.8, seed: int = 42):
    """
    For each *_filtered.jsonl file in processed_dir:
    1. Load the JSONL file
    2. Perform event-based split
    3. Verify no leakage (crash with error if leakage detected)
    4. Save to output_dir/train_<task>.jsonl and output_dir/val_<task>.jsonl
    5. Print summary table per task
    
    Final summary printed at end:
    | Task | Train Samples | Val Samples | Num Events Train | Num Events Val |
    
    Save a split_report.json to output_dir with all statistics for reference.
    """
    if not os.path.exists(processed_dir):
        print(f"Error: Processed directory not found at: {processed_dir}")
        print("Run filter_disasters.py first to generate the data.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    
    for f_name in os.listdir(processed_dir):
        if f_name.endswith("_filtered.jsonl"):
            task_name = f_name.replace("_filtered.jsonl", "")
            file_path = os.path.join(processed_dir, f_name)
            
            print(f"\nProcessing {f_name}...")
            data = load_jsonl(file_path)
            
            if not data:
                continue
                
            train_data, val_data = event_based_split(data, train_ratio=train_ratio, seed=seed)
            
            if not verify_no_event_leakage(train_data, val_data):
                raise RuntimeError("Data leakage detected. Aborting split pipeline.")
                
            train_out = os.path.join(output_dir, f"train_{task_name}.jsonl")
            val_out = os.path.join(output_dir, f"val_{task_name}.jsonl")
            
            try:
                with open(train_out, 'w', encoding='utf-8') as f:
                    for record in train_data:
                        f.write(json.dumps(record) + "\n")
                with open(val_out, 'w', encoding='utf-8') as f:
                    for record in val_data:
                        f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"Error writing output files for {task_name}: {e}")
                
            train_events = len(set([r.get("event_id") for r in train_data]))
            val_events = len(set([r.get("event_id") for r in val_data]))
            
            summary.append({
                "Task": task_name,
                "Train Samples": len(train_data),
                "Val Samples": len(val_data),
                "Num Events Train": train_events,
                "Num Events Val": val_events
            })
            
    if not summary:
        print("No *_filtered.jsonl files found to split.")
        return
        
    print("\n" + "="*70)
    print("SPLIT PIPELINE SUMMARY")
    print(f"{'Task':<20} | {'Train Samples':<15} | {'Val Samples':<13} | {'Num Events Train':<16} | {'Num Events Val'}")
    print("-" * 70)
    for s in summary:
        print(f"{s['Task']:<20} | {s['Train Samples']:<15} | {s['Val Samples']:<13} | {s['Num Events Train']:<16} | {s['Num Events Val']}")
    print("="*70)
    
    report_path = os.path.join(output_dir, "split_report.json")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved split report to {report_path}")
    except Exception as e:
        print(f"Error saving split report: {e}")

if __name__ == "__main__":
    try:
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("configs/config.yaml not found. Please run this script from the project root.")
        exit(1)
        
    run_split_pipeline(
        processed_dir=config["paths"]["processed_dir"],
        output_dir=config["paths"]["processed_dir"],
        train_ratio=0.8,
        seed=config["project"]["seed"]
    )
    
    print("\nSplit complete. Files saved to:", config["paths"]["processed_dir"])
    print("Bench_set is the test set \u2014 use it only for final evaluation.")
