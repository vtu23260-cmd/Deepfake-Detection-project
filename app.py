import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# =====================
# LOAD MODEL
# =====================
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded!")

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================
# HEATMAP (IMPROVED)
# =====================
def generate_heatmap(image):
    image = image.convert("RGB")

    img = transform(image).unsqueeze(0)
    img.requires_grad_(True)

    output = model(img)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    gradients = img.grad

    if gradients is None:
        return np.zeros((224, 224), dtype=np.uint8)

    gradients = gradients.detach().cpu().numpy()[0]

    heatmap = np.mean(gradients, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    return heatmap

# =====================
# OVERLAY HEATMAP (PRO LOOK)
# =====================
def overlay_heatmap(image, heatmap):
    image = np.array(image.resize((224, 224)))

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay

# =====================
# SMART EXPLANATION
# =====================
def explain_heatmap(heatmap):
    h, w = heatmap.shape

    # Center region
    center = heatmap[h//4:3*h//4, w//4:3*w//4]

    # Edge regions (calculate separately instead of concatenating)
    top = heatmap[:h//4, :]
    bottom = heatmap[3*h//4:, :]
    left = heatmap[:, :w//4]
    right = heatmap[:, 3*w//4:]

    # Compute averages safely
    center_score = np.mean(center)
    edge_score = np.mean([np.mean(top), np.mean(bottom), np.mean(left), np.mean(right)])

    if center_score > edge_score + 15:
        return "The model detected strong anomalies in the main subject area (face/object), suggesting AI-generated inconsistencies."
    elif edge_score > center_score + 15:
        return "The model found irregularities in the background (lighting, textures, or edges), often seen in AI-generated images."
    else:
        return "The model detected subtle inconsistencies across the image, indicating possible AI generation."

# =====================
# PREDICTION FUNCTION
# =====================
def predict(image):
    try:
        if image is None:
            return "Please upload an image", None

        # Ensure RGB
        image = image.convert("RGB")

        img = transform(image).unsqueeze(0)
        img.requires_grad_(True)

        # Forward pass (NO torch.no_grad here ❗)
        output = model(img)
        probs = F.softmax(output, dim=1)[0]

        # Generate heatmap
        heatmap = generate_heatmap(image)

        # Overlay
        overlay = overlay_heatmap(image, heatmap)

        explanation = explain_heatmap(heatmap)

        result_text = (
            f"🧠 Result: {'AI Generated' if probs[1] > probs[0] else 'Real Image'}\n\n"
            f"📊 AI Confidence: {probs[1]:.2f}\n"
            f"📊 Real Confidence: {probs[0]:.2f}\n\n"
            f"🔍 Explanation:\n{explanation}"
        )

        return result_text, overlay

    except Exception as e:
        return f"❌ Error occurred: {str(e)}", None

# =====================
# UI (FINAL POLISH)
# =====================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📤 Upload Image"),
    outputs=[
        gr.Textbox(label="📊 Result & Explanation"),
        gr.Image(label="🔥 AI Focus Visualization")
    ],
    title="🧠 AI Image Detector PRO",
    description="Detect whether an image is AI-generated or real with visual explanation.",
    theme="default"
)

# =====================
# RUN
# =====================
demo.launch()