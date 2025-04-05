import json
from ultralytics import YOLO

model = YOLO("training/weights/best.pt")

results = model.predict(
    # source="/data/tjf5667/datasets/PCB_DATASET/output/images/train/",
    source="/data/tjf5667/datasets/PCB_DATASET/output/images/val/",
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
    prompt = []
    explanations = []
    
    if result.boxes.cls.numel() == 0:
        output.append({
            "prompt": "",
            "explanation": "No defects detected."
        })
        continue

    for box in result.boxes:
        defect_type = result.names[int(box.cls)]
        coordinates = [round(coord, 2) for coord in box.xywh[0].tolist()]
        confidence = round(float(box.conf)*100, 2)
        prompt.append(f"{defect_type}, {coordinates}, {confidence}")

        custom_description = class_descriptions.get(int(box.cls), "Unknown defect type.")
        custom_name = class_names.get(int(box.cls), "Unknown defect name.")
        explanations.append(f"{custom_name} detected at ({coordinates[0]}, {coordinates[1]}) with {confidence}% confidence - {custom_description}.")

    explanation = "\n".join(explanations)
    prompt = '\n'.join(prompt)

    data = {
        "prompt": prompt,
        "explanation": explanation
    }
    output.append(data)

# output_file = "/data/tjf5667/CSE587/train.json"
output_file = "/data/tjf5667/CSE587/val.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Results saved to {output_file}")