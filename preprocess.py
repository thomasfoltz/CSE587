import json
from ultralytics import YOLO

model = YOLO("training/weights/best.pt")

results = model.predict(
    # source="../datasets/PCB_DATASET/output/images/train/",
    source="../datasets/PCB_DATASET/output/images/val/",
)

class_descriptions = {
    0: "indicating a required drilled hole is absent, leading to connectivity issues",
    1: "indicating a jagged notche or perforation on the edge of the PCB",
    2: "indicating a break in the conductive path",
    3: "indicating an unintended connection between two conductive paths",
    4: "indicating a thin, unintended copper trace extending from the circuit, potentially causing interference or shorts",
    5: "indicating an unwanted copper remnant on the PCB surface, which could lead to electrical faults or shorts"
}

class_names = {
    0: "Missing Hole",
    1: "Mouse Bite",
    2: "Open Circuit",
    3: "Short Circuit",
    4: "Spur",
    5: "Spurious Copper"
}

output = []
for result in results:
    for box in result.boxes:
        defect_class = int(box.cls)
        x = round(box.xywh[0][0].item(), 2)  # Extract X
        y = round(box.xywh[0][1].item(), 2)  # Extract Y
        confidence = f"{round(float(box.conf) * 100, 2)}%"  # Add %
        
        # Use human-readable class names
        defect_name = class_names.get(defect_class, "Unknown Defect")
        description = class_descriptions.get(defect_class, "")
        
        # Create individual training examples per defect
        prompt = f"Describe defect: {defect_name}, ({x}, {y}), {confidence}"
        explanation = f"{defect_name} detected at ({x}, {y}) with {confidence} confidence, {description}."
        
        output.append({"prompt": prompt, "explanation": explanation})


# output_file = "./train.json"
output_file = "./val.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Results saved to {output_file}")