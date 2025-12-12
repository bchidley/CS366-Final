import torch
import os
import csv
import gc # <--- ADDED THIS
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- CONFIGURATION ---
INPUT_ROOT_DIR = "/home/bchidley/gpu_data/overlays/" 
OUTPUT_CSV = "/home/bchidley/CS366-Final/labels/car_counts.csv"

# --- 1. Load Model ---
print("Loading model...")
model = build_sam3_image_model().to('cuda')
checkpoint = torch.load('/home/bchidley/sam3/sam3_weights/sam3.pt', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
processor = Sam3Processor(model)

# --- 2. Create CSV ---
print(f"Starting scan of {INPUT_ROOT_DIR}...")

# Open in append mode 'a' if you want to resume, or 'w' to restart. 
# Since you crashed, let's stick to 'w' (restart) or manually edit the CSV later.
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Car_Count', 'Full_Path'])

    total_processed = 0
    
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for filename in sorted(files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                full_path = os.path.join(root, filename)
                
                # Load Image
                image = Image.open(full_path).convert("RGB")

                # Run Inference
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    inference_state = processor.set_image(image)
                    output = processor.set_text_prompt(state=inference_state, prompt="car")

                if isinstance(output, list):
                    output = output[0]

                count = 0
                if 'boxes' in output:
                    count = len(output['boxes'])
                
                # Write to CSV
                writer.writerow([filename, count, full_path])
                
                # --- MEMORY FIX STARTS HERE ---
                # 1. Delete the heavy variables from Python memory
                del output
                del inference_state
                del image
                
                # 2. Force Python to run garbage collection
                gc.collect()
                
                # 3. Force PyTorch to empty the GPU cache
                torch.cuda.empty_cache()
                # --- MEMORY FIX ENDS HERE ---

                total_processed += 1

                print(f"Processed {total_processed} images... (Current: {filename})-- Car Count: {count}")

print(f"Done! Labeled {total_processed} images.")