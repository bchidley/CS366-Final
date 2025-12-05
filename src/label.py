import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

# Load the model
model = build_sam3_image_model().to('cuda')
# Load the checkpoint
checkpoint = torch.load('/home/bchidley/sam3/sam3_weights/sam3.pt', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
processor = Sam3Processor(model)
# Load an image
image = Image.open("/home/bchidley/gpu_data/final/Tennis2/Tennis2_day_000209.jpg")
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="car")

img0 = Image.open("/home/bchidley/gpu_data/final/Tennis2/Tennis2_day_000209.jpg")
fig = plot_results(img0, inference_state)
# Save the plot to local
plt.savefig("/home/bchidley/sam3/test_result/sam3_results.png", bbox_inches='tight', dpi=150)
plt.close()
