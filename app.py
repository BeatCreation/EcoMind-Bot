# app.py

import gradio as gr
from classifier_module import predict_image
from chatbot_module import get_bot_response

# ------------------------
# Waste Classification Logic with Eco Tip
# ------------------------
def classify_waste(image):
    if image is None:
        return "Please upload an image.", None

    result = predict_image(image)
    category = result.split("(")[0].strip()

    tips = {
        "Plastic": "Reduce single-use plastics. Try reusable bottles and containers.",
        "Organic": "Compost it! Organic waste enriches soil and reduces landfill load.",
        "Metal": "Recycle at authorized metal scrap centers to reduce mining impacts.",
        "Glass": "Glass is 100% recyclable. Rinse and recycle without labels.",
        "Paper": "Avoid contamination. Keep paper dry and clean for effective recycling.",
        "E-waste": "Dispose e-waste at certified recycling facilities. Don’t toss it in bins!"
    }

    tip = tips.get(category, "Use local recycling guidelines to dispose of this item responsibly.")
    return f"🧠 Predicted Category: {result}", f"🌿 Eco Tip: {tip}"

# ------------------------
# Chatbot Response
# ------------------------
def chatbot_response(message):
    print("Frontend sent message:", repr(message))
    response = get_bot_response(message)
    print("GPT response:", response)
    return response


# ------------------------
# Waste Classifier UI
# ------------------------
classifier_ui = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="📸 Upload a Waste Image (PNG/JPG)"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Eco Tip")
    ],
    title="♻️ EcoMind Waste Classifier",
    description="Upload a photo of waste for classification and get an eco-friendly disposal tip. 🌍",
    theme="default"
)

# ------------------------
# Chatbot UI
# ------------------------
chatbot_ui = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me about recycling or sustainability..."),
    outputs=gr.Textbox(label="EcoMind Bot"),
    title="💬 Chat with EcoMind",
    description="Ask anything about sustainability, recycling, or waste management.",
    theme="default"
)

# ------------------------
# Combined Tabbed UI
# ------------------------
app = gr.TabbedInterface(
    [classifier_ui, chatbot_ui],
    ["📸 Waste Classifier", "💬 Chat with EcoMind"],
    css="""
    body { background-color: #f0f4f7; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #2b6777; }
    .gr-box { background-color: #eaf7f3 !important; border-radius: 12px; padding: 16px; }
    """
)

app.launch()
