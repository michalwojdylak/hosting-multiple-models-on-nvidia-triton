import os
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

MODELS = [
    ("resnet50", "microsoft/resnet-50"),
    ("convnext_tiny", "facebook/convnext-tiny-224"),
    ("mobilenet_v2", "google/mobilenet_v2_1.0_224"),
]

OUTPUT_DIR = "models"


def export_to_onnx(name: str, hf_model: str):
    print(f"Loading {hf_model}...")

    processor = AutoImageProcessor.from_pretrained(hf_model)
    model = AutoModelForImageClassification.from_pretrained(hf_model)
    model.eval()

    size = processor.size
    if "shortest_edge" in size:
        image_size = size["shortest_edge"]
    elif "height" in size:
        image_size = size["height"]
    else:
        image_size = 224

    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Exporting to {output_path}...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

    print(f"✓ Exported {output_path}")


def main():
    for name, hf_model in MODELS:
        export_to_onnx(name, hf_model)


if __name__ == "__main__":
    main()