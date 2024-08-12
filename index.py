from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import genai
import textwrap
from IPython.display import Markdown

app = Flask(__name__)

# Initialize model components
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
text_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set up the device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(device)

# Define generation parameters
captioning_config = {"max_length": 16, "num_beams": 4}
genai_model = genai.GenerativeModel('gemini-pro')

# Configure GenAI API
genai.configure(api_key='KEY')  # Replace 'KEY' with your actual API key

def format_to_markdown(text):
    """Convert text to a markdown format with bullet points."""
    text = text.replace('•', '* ')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def generate_caption_from_image(image_path):
    """Generate captions for an image using the captioning model."""
    img = Image.open(image_path).convert("RGB")
    image_tensor = image_processor(images=[img], return_tensors="pt").pixel_values.to(device)

    generated_ids = captioning_model.generate(image_tensor, **captioning_config)
    caption = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return caption

def create_instagram_caption(image_path, mood):
    """Generate Instagram captions based on the image and mood."""
    image_caption = generate_caption_from_image(image_path)
    prompt = f"Create Instagram captions with emojis and hashtags. The image shows {image_caption}, and the mood is {mood}."
    ai_response = genai_model.generate_content(prompt)
    
    return format_to_markdown(ai_response.text)

@app.route('/generate_caption', methods=['POST'])
def handle_caption_generation():
    """API endpoint to generate captions based on image and mood."""
    data = request.get_json()
    image_path = data['image_path']
    mood = data['mood_category']
    
    caption = create_instagram_caption(image_path, mood)
    return jsonify({"result": caption.data})  # .data attribute is needed to extract text from Markdown

if __name__ == '__main__':
    app.run(debug=True)
