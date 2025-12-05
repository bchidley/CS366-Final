import torch
import os
import csv
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
from transformers import Sam3Processor, Sam3Model
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

ROOT_DIR = "/home/bchidley/gpu_data/final"
OUTPUT_DIR = "/home/bchidley/sam3/test_result"
CSV_PATH = os.path.join(OUTPUT_DIR, "car_counts.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load the model
model = build_sam3_image_model().to('cuda')
checkpoint = torch.load('/home/bchidley/sam3/sam3_weights/sam3.pt', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
processor = Sam3Processor(model)

def normalize_to_1000(boxes, img_width, img_height):
    """
    Scales bounding boxes from absolute pixels to the 0-1000 range
    required by this specific SAM3 implementation.
    """
    norm_boxes = []
    for box in boxes:
        # Original format: [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = box
        
        # Scale to 1000
        nx1 = int((x1 / img_width) * 1000)
        ny1 = int((y1 / img_height) * 1000)
        nx2 = int((x2 / img_width) * 1000)
        ny2 = int((y2 / img_height) * 1000)
        
        norm_boxes.append([nx1, ny1, nx2, ny2])
    return norm_boxes

#-----------------------Bounding Boxes for each folder-----------------------#
# Tennis2 (1920x1080)
tennis2_boxes = [
    [408, 386, 1394, 702],
    [631, 331, 1433, 410],
    [806, 295, 1528, 348],
    [1009, 251, 1684, 349],
    [1179, 223, 1786, 299],
    [1299, 196, 1825, 245],
    [1389, 156, 1849, 217],
    [573, 346, 672, 404]
]
tennis2_label = [[1,1,1,1,1,1,1,1]]

# Tennis1 (2592x1944)
tennis1_boxes = [
    [1120, 387, 2058, 566],
    [1243, 350, 2187, 421],
    [1486, 321, 2180, 392],
    [1681, 302, 2196, 373],
    [1912, 279, 2226, 357]
]
tennis1_label = [[1,1,1,1,1]]

# North1 (2592x1944)
north1_boxes = [
    [0, 579, 2584, 1937],
    [2188, 557, 2488, 591],
    [2171, 532, 2391, 585],
    [2162, 510, 2304, 587],
    [1750, 464, 2128, 629],
    [1773, 453, 2034, 485],
    [1771, 441, 1959, 485],
    [1718, 424, 1871, 610],
    [0, 408, 1768, 651],
    [1377, 393, 1594, 462],
    [1166, 385, 1512, 410]
]
north1_label = [[1,1,1,1,1,1,1,1,1,1,1]]
# North2 (2592x1944)
north2_boxes = [
    [0, 645, 2590, 1941],
    [51, 622, 1663, 705],
    [1381, 368, 1988, 679],
    [1972, 413, 2121, 651],
    [2115, 585, 2589, 679],
    [2110, 459, 2155, 614],
    [2158, 475, 2215, 610],
    [2210, 487, 2288, 615],
    [2272, 504, 2400, 605],
    [1196, 382, 1393, 663],
    [1072, 391, 1198, 753],
    [917, 406, 1079, 713],
    [850, 420, 930, 658],
    [733, 440, 862, 657],
    [665, 450, 740, 683],
    [566, 467, 669, 719],
    [63, 593, 585, 682],
    [154, 584, 598, 614],
    [221, 570, 582, 598],
    [308, 542, 589, 569],
    [367, 524, 603, 553],
    [466, 498, 601, 534]
]
north2_label = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

# Root1 and Root2 (2592x1944)
root_boxes = [
    [0, 645, 2590, 1943],
    [1372, 232, 2588, 816],
    [1061, 306, 1400, 666],
    [859, 354, 1134, 679],
    [768, 376, 903, 719],
    [729, 385, 779, 643],
    [669, 395, 736, 665],
    [530, 428, 703, 689],
    [463, 443, 546, 665],
    [0, 588, 333, 661]
]
root_label = [[1,1,1,1,1,1,1,1,1,1]]
#NO BBoxes for RootStaff or MMM1/MMM2



#-----------Done with Bounding Boxes for each folder-----------#



# Image Process Loop
with open(CSV_PATH, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "count"])

    for folder in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # Set up prompts for this folder
        current_raw_boxes = []
        use_bbox = False

        if folder == "Tennis2":
            current_raw_boxes = tennis2_boxes
            use_bbox = True
            lebron = tennis2_label
        elif folder == "Tennis1":
            current_raw_boxes = tennis1_boxes
            use_bbox = True
            lebron = tennis1_label
        elif folder in ["MMM1", "MMM2"]:
            use_bbox = False
        elif folder == "North1":
            current_raw_boxes = north1_boxes
            use_bbox = True
            lebron = north1_label
        elif folder == "North2":
            current_raw_boxes = north2_boxes
            use_bbox = True
            lebron = north2_label
        elif folder in ["Root1", "Root2"]:
            current_raw_boxes = root_boxes
            use_bbox = True
            lebron = root_label
        elif folder == "RootStaff":
            use_bbox = False
        else:
            continue
        
        print(f"Processing folder: {folder}")

        for frame in os.listdir(folder_path):
            #frame = image
            #current_raw_boxes = unnormalized boxes for that folder
            #lebron = labels for that folder
            #text will be "car"

            inputs = processor(
                images=frame,
                text="car",
                input_boxes=current_raw_boxes if use_bbox else None,
                input_boxes_labels=lebron, 
                return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
