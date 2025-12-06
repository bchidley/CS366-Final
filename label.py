import torch
#################################### For Image #################################
###
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # <--- Added for drawing boxes

# Specific file path
INPUT_ROOT_DIR = "/home/bchidley/gpu_data/overlays/North1_North1_day_000100.jpg" 
OUTPUT_CSV = "/home/bchidley/CS366-Final/labels/car_counts.csv"

# Load the model
model = build_sam3_image_model().to('cuda')
checkpoint = torch.load('/home/bchidley/sam3/sam3_weights/sam3.pt', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
processor = Sam3Processor(model)

image = Image.open(INPUT_ROOT_DIR).convert("RGB")

with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="car")

# --- CUSTOM DRAWING CODE STARTS HERE ---

# 1. Setup the plot
plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

# 2. Extract boxes safely
if isinstance(output, list):
    output = output[0]

# Check for 'boxes' (likely) or 'pred_boxes' (fallback)
if 'boxes' in output:
    boxes = output['boxes']
elif 'pred_boxes' in output:
    boxes = output['pred_boxes']
else:
    boxes = []

# Move from GPU to CPU/Numpy for plotting
if hasattr(boxes, 'detach'):
    boxes = boxes.detach().cpu().numpy()

# 3. Loop and draw GREEN boxes only (No text/ID)
for box in boxes:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), width, height, 
        linewidth=2, 
        edgecolor='#00FF00', # Bright Green
        facecolor='none'     # Transparent center
    )
    ax.add_patch(rect)

# 4. Save
plt.axis('off')
plt.savefig("/home/bchidley/sam3/test_result/N1D.jpg", bbox_inches='tight', pad_inches=0, dpi=150)
plt.close()

print("scp bchidley@150.209.91.70:/home/bchidley/sam3/test_result/N1D.jpg ~/Downloads/")