import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model name and debug mode from environment variables
model_name = os.getenv("MODEL_NAME", "t5-small")
debug_mode = os.getenv("DEBUG", "False").lower() == "true"

# Initialize the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    logger.info(f"Successfully initialized model and tokenizer: {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize model or tokenizer: {str(e)}")
    model = None
    tokenizer = None

def generate_paraphrase(text):
    """Generate paraphrase using T5 model."""
    try:
        # Prepare the input text
        input_text = f"paraphrase: {text}"
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate output
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            early_stopping=True,
            num_return_sequences=1
        )
        
        # Decode the output
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated paraphrase: {paraphrased_text}")
        
        return paraphrased_text
        
    except Exception as e:
        logger.error(f"Error in generate_paraphrase: {str(e)}")
        raise

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Handle paraphrasing requests."""
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not initialized")
        return jsonify({"error": "Model is not initialized properly"}), 500

    # Validate request
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        text = data.get('text')
        if not text or not isinstance(text, str):
            return jsonify({"error": "No text provided or invalid text format"}), 400
        
        if len(text.strip()) == 0:
            return jsonify({"error": "Empty text provided"}), 400

        # Generate paraphrase
        paraphrased_text = generate_paraphrase(text)
        
        if not paraphrased_text:
            return jsonify({"error": "Failed to generate paraphrased text"}), 500
        
        return jsonify({"paraphrased_text": paraphrased_text})

    except Exception as e:
        logger.error(f"Error during paraphrasing: {str(e)}", exc_info=True)
        return jsonify({"error": f"Paraphrasing failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status."""
    status = {
        "status": "healthy",
        "model": {
            "name": model_name,
            "initialized": model is not None and tokenizer is not None
        }
    }
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
