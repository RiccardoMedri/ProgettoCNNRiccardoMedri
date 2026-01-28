import os


#Carica un file di annotazioni VisDrone e restituisce boxes XYXY + labels rimappate
def load_visdrone_annotations(ann_path, class_map):
    boxes = []
    labels = []
    if not os.path.exists(ann_path):
        return boxes, labels

    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, _, category = parts[:6]
            category = int(category)
            if category <= 0 or category not in class_map:
                continue
            mapped = class_map[category]
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            boxes.append([x, y, x + w, y + h])
            labels.append(mapped)

    return boxes, labels
