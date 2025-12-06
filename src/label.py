import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import matplotlib.patches as patches 

DIR = "/home/bchidley/gpu_data/final/RootStaff/RootStaff_night_000050.jpg"

# Load the model
model = build_sam3_image_model().to('cuda')
checkpoint = torch.load('/home/bchidley/sam3/sam3_weights/sam3.pt', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
processor = Sam3Processor(model)

# Load an image
image = Image.open(DIR).convert("RGB")

with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="car")

################### FIX STARTS HERE ###################

plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

# FIX: Check for the correct key ('boxes' vs 'pred_boxes')
# We also check if output is a list (batch format), though the error suggested it is a dict.
if isinstance(output, list):
    output = output[0]

if 'boxes' in output:
    boxes = output['boxes']       # This is the most likely key
elif 'pred_boxes' in output:
    boxes = output['pred_boxes']  # The fallback
else:
    # Print available keys if both fail so we can debug
    print(f"Error: Could not find boxes. Available keys: {output.keys()}")
    boxes = []

if hasattr(boxes, 'detach'): # Check if it's a tensor
    boxes = boxes.detach().cpu().numpy()

for box in boxes:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), width, height, 
        linewidth=2, 
        edgecolor='green', 
        facecolor='none'
    )
    ax.add_patch(rect)

plt.axis('off')
plt.savefig("/home/bchidley/sam3/test_result/RootStaff_Night.jpg", bbox_inches='tight', dpi=150)
plt.close()