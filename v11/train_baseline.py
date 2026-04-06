from ultralytics import YOLO
import sys


def parse_args():
    batch = 4
    device = "0"
    epochs = 400
    imgsz = 1280
    workers = 0
    amp = False
    data = "./data/data.yaml"
    project = "runs/detect_baseline"
    patience = 20
    name = "baseline_stage2"

    for arg in sys.argv[1:]:
        if arg.startswith("--batch="):
            batch = int(arg.split("=")[1])
        elif arg.startswith("--device="):
            device = arg.split("=")[1]
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=")[1])
        elif arg.startswith("--imgsz="):
            imgsz = int(arg.split("=")[1])
        elif arg.startswith("--workers="):
            workers = int(arg.split("=")[1])
        elif arg.startswith("--amp="):
            amp = arg.split("=")[1].lower() == "true"
        elif arg.startswith("--data="):
            data = arg.split("=")[1]
        elif arg.startswith("--project="):
            project = arg.split("=")[1]
        elif arg.startswith("--patience="):
            patience = int(arg.split("=")[1])
        elif arg.startswith("--name="):
            name = arg.split("=")[1]

    return {
        "batch": batch,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "workers": workers,
        "amp": amp,
        "data": data,
        "project": project,
        "patience": patience,
        "name": name,
    }


args = parse_args()
model = YOLO("yolo11m.pt")

overrides = {
    "data": args["data"],
    "epochs": args["epochs"],
    "batch": args["batch"],
    "imgsz": args["imgsz"],
    "scale": 0.9,
    "mosaic": 0.6,
    "mixup": 0.15,
    "copy_paste": 0.4,
    "device": args["device"],
    "workers": args["workers"],
    "amp": args["amp"],
    "patience": args["patience"],
    "project": args["project"],
    "name": args["name"],
    "exist_ok": True,
}

results = model.train(**overrides)

metrics = model.val(
    data=args["data"],
    project=args["project"],
    name=args["name"],
    exist_ok=True,
)
