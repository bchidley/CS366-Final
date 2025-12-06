import os
import shutil

# --- CONFIGURATION ---
# 1. Where do you want the samples to go so you can download them?
EXPORT_DIR = "/home/bchidley/CS366-Final/professor_samples"

# 2. Your list of files (Filename, Count, FullPath)
csv_data = """
MMM1_MMM1_day_000257.jpg,45,/home/bchidley/gpu_data/overlays/MMM1_MMM1_day_000257.jpg
North1_North1_day_000087.jpg,46,/home/bchidley/gpu_data/overlays/North1_North1_day_000087.jpg
North2_North2_day_000356.jpg,46,/home/bchidley/gpu_data/overlays/North2_North2_day_000356.jpg
North2_North2_night_000081.jpg,48,/home/bchidley/gpu_data/overlays/North2_North2_night_000081.jpg
Root1_Root1_day_000143.jpg,65,/home/bchidley/gpu_data/overlays/Root1_Root1_day_000143.jpg
Root1_Root1_night_000108.jpg,60,/home/bchidley/gpu_data/overlays/Root1_Root1_night_000108.jpg
RootStaff_RootStaff_day_000461.jpg,8,/home/bchidley/gpu_data/overlays/RootStaff_RootStaff_day_000461.jpg
RootStaff_RootStaff_night_000090.jpg,1,/home/bchidley/gpu_data/overlays/RootStaff_RootStaff_night_000090.jpg
Tennis1_Tennis1_day_000744.jpg,10,/home/bchidley/gpu_data/overlays/Tennis1_Tennis1_day_000744.jpg
Tennis1_Tennis1_night_000150.jpg,9,/home/bchidley/gpu_data/overlays/Tennis1_Tennis1_night_000150.jpg
Tennis2_Tennis2_day_000313.jpg,36,/home/bchidley/gpu_data/overlays/Tennis2_Tennis2_day_000313.jpg
"""

# --- EXECUTION ---
print(f"Creating export directory: {EXPORT_DIR}")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Split the data into lines and process
lines = csv_data.strip().split('\n')

success_count = 0

for line in lines:
    parts = line.split(',')
    
    # Safety check: make sure the line has 3 parts
    if len(parts) < 3:
        continue
        
    filename = parts[0].strip()
    src_path = parts[2].strip() # The full path is the 3rd item
    dst_path = os.path.join(EXPORT_DIR, filename)

    try:
        # 1. COPY to export folder
        shutil.copy(src_path, dst_path)
        
        # 2. DELETE from source
        os.remove(src_path)
        
        print(f"Moved & Deleted: {filename}")
        success_count += 1
        
    except FileNotFoundError:
        print(f"Skipping (File not found): {src_path}")
    except Exception as e:
        print(f"Error on {filename}: {e}")

print("------------------------------------------------")
print(f"Done! Successfully moved {success_count} files.")
print(f"You can download them from: {EXPORT_DIR}")