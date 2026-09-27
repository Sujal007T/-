import os
import json
import argparse
import yaml

def inspect_annotation_file(annotation_path: str) -> dict:
    """
    Opens an annotation JSON file and prints:
    1. Total number of records
    2. All keys present in the first record
    3. All unique disaster_type values (or equivalent field)
    4. All unique event_id values (or equivalent field)
    5. Sample of 3 records in pretty-printed JSON format
    
    Returns the raw data list.
    Call this FIRST before running any filtering.
    """
    if not os.path.exists(annotation_path):
        print(f"Error: File not found at {annotation_path}")
        return []
        
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {annotation_path}: {e}")
        return []
        
    print(f"1. Total number of records: {len(data)}")
    
    if not data:
        print("Data is empty.")
        return data
        
    print(f"2. All keys present in the first record: {list(data[0].keys())}")
    
    disaster_types = set()
    event_ids = set()
    
    for record in data:
        disaster_types.add(record.get("disaster_type", "UNKNOWN"))
        event_ids.add(record.get("event_id", "UNKNOWN"))
        
    print(f"3. All unique disaster_type values: {list(disaster_types)}")
    print(f"4. All unique event_id values: {list(event_ids)}")
    
    print("5. Sample of 3 records:")
    sample = data[:min(3, len(data))]
    print(json.dumps(sample, indent=4))
    
    return data

def filter_disaster_types(data: list, selected_types: list, 
                           disaster_type_field: str = "disaster_type") -> list:
    """
    Filters records to only those matching selected disaster types.
    Case-insensitive matching.
    Prints count before and after filtering.
    Prints warning if any selected_type has zero matching records.
    """
    selected_lower = [t.lower() for t in selected_types]
    filtered_data = []
    counts = {t: 0 for t in selected_lower}
    
    for record in data:
        d_type = record.get(disaster_type_field, "").lower()
        if d_type in selected_lower:
            filtered_data.append(record)
            counts[d_type] += 1
            
    print(f"Records before filtering: {len(data)}")
    print(f"Records after filtering: {len(filtered_data)}")
    
    for t in selected_lower:
        if counts[t] == 0:
            print(f"Warning: Selected disaster type '{t}' has zero matching records.")
            
    return filtered_data

def save_to_jsonl(data: list, output_path: str, task_name: str):
    """
    Saves a list of dicts to JSONL format (one JSON per line).
    Adds a "task" field to each record with value task_name.
    Prints: total records written, file size, output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in data:
                record["task"] = task_name
                f.write(json.dumps(record) + "\n")
                
        file_size = os.path.getsize(output_path)
        print(f"Saved {len(data)} records to {output_path} (Size: {file_size} bytes).")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

def run_filter_pipeline(disasterm3_root: str, output_dir: str,
                         selected_disaster_types: list = None):
    """
    Main pipeline:
    1. Checks disasterm3_root exists — if not, prints instructions and exits
    2. Finds all annotation JSON files in Instruct_set/annotations/
    3. For each annotation file:
       a. Calls inspect_annotation_file (first run only, or if --inspect flag)
       b. Calls filter_disaster_types
       c. Saves filtered data to output_dir/<task_name>_filtered.jsonl
    4. Prints summary table:
       | Task | Total Records | After Filter | Disaster Types |
    """
    if selected_disaster_types is None:
        selected_disaster_types = ["flooding", "earthquake", "hurricane", "wildfire", "landslide"]
        
    if not os.path.exists(disasterm3_root):
        print(f"Error: DisasterM3 root path not found at: {disasterm3_root}")
        print("Please upload/download the DisasterM3 dataset to Google Drive (or update config.yaml paths).")
        return
        
    ann_dir = os.path.join(disasterm3_root, "Instruct_set", "annotations")
    if not os.path.exists(ann_dir):
        print(f"Error: Annotations directory not found at: {ann_dir}")
        print("Check your DisasterM3 directory structure.")
        return
        
    print(f"Running filter pipeline on {ann_dir}...")
    
    summary = []
    
    for f_name in os.listdir(ann_dir):
        if f_name.endswith(".json"):
            file_path = os.path.join(ann_dir, f_name)
            task_name = f_name.replace(".json", "")
            
            print(f"\nProcessing {f_name}...")
            data = inspect_annotation_file(file_path)
            
            if not data:
                continue
                
            filtered_data = filter_disaster_types(data, selected_disaster_types)
            
            out_file = os.path.join(output_dir, f"{task_name}_filtered.jsonl")
            save_to_jsonl(filtered_data, out_file, task_name)
            
            # Find disaster types in filtered data
            found_types = list(set([r.get("disaster_type", "UNKNOWN") for r in filtered_data]))
            
            summary.append({
                "Task": task_name,
                "Total Records": len(data),
                "After Filter": len(filtered_data),
                "Disaster Types": len(found_types)
            })
            
    if not summary:
        print("No annotation files found or processed.")
        return
        
    print("\n" + "="*60)
    print("FILTER PIPELINE SUMMARY")
    print(f"{'Task':<25} | {'Total Records':<13} | {'After Filter':<12} | {'Disaster Types'}")
    print("-" * 60)
    for s in summary:
        print(f"{s['Task']:<25} | {s['Total Records']:<13} | {s['After Filter']:<12} | {s['Disaster Types']}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                        help="Print annotation file structure and exit")
    parser.add_argument("--types", nargs="+", 
                        default=["flooding", "earthquake", "hurricane", 
                                 "wildfire", "landslide"])
    args = parser.parse_args()
    
    try:
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("configs/config.yaml not found. Please run this script from the project root.")
        exit(1)
    
    if args.inspect:
        # Just inspect and print, don't filter
        ann_dir = os.path.join(config["paths"]["disasterm3_root"], 
                               "Instruct_set", "annotations")
        if not os.path.exists(ann_dir):
            print(f"Annotations directory not found at: {ann_dir}. Ensure dataset is downloaded.")
        else:
            for f_name in os.listdir(ann_dir):
                if f_name.endswith(".json"):
                    print(f"\n{'='*50}\nInspecting: {f_name}")
                    inspect_annotation_file(os.path.join(ann_dir, f_name))
    else:
        run_filter_pipeline(
            disasterm3_root=config["paths"]["disasterm3_root"],
            output_dir=config["paths"]["processed_dir"],
            selected_disaster_types=args.types
        )
