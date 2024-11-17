import os
from flask import Flask, request, jsonify
from transformers import pipeline
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

# Initialize the paraphrasing model with specific configuration
try:
    paraphraser = pipeline(
        task="text2text-generation",
        model=model_name,
        framework="pt",  # Explicitly specify PyTorch
        device=-1  # Use CPU, change to 0 for GPU if available
    )
    logger.info(f"Successfully initialized model: {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    paraphraser = None

def preprocess_text(text):
    """Prepare the input text for the T5 model."""
    # T5 expects a specific prefix for different tasks
    return f"paraphrase: {text}"

def postprocess_text(model_output):
    """Clean up the model output."""
    if not model_output or not isinstance(model_output, list):
        return None
    
    # Extract the generated text from the model output
    try:
        generated_text = model_output[0].get('generated_text', '').strip()
        return generated_text if generated_text else None
    except (IndexError, AttributeError) as e:
        logger.error(f"Error in postprocessing: {str(e)}")
        return None

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Handle paraphrasing requests."""
    if paraphraser is None:
        logger.error("Paraphrasing model not initialized")
        return jsonify({"error": "Model is not initialized properly"}), 500

    # Validate request
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    text = data.get('text')
    if not text or not isinstance(text, str):
        return jsonify({"error": "No text provided or invalid text format"}), 400
    
    if len(text.strip()) == 0:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        # Log input text for debugging
        logger.info(f"Processing input text: {text[:100]}...")  # Log first 100 chars
        
        # Preprocess the input text
        processed_input = preprocess_text(text)
        
        # Generate paraphrased text with specific parameters
        response = paraphraser(
            processed_input,
            max_length=min(len(text.split()) * 2, 512),  # Dynamic length with cap
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,  # Control randomness
            top_k=50,  # Limit vocabulary for better quality
            top_p=0.95,  # Nucleus sampling
            early_stopping=True
        )
        
        # Log raw model output for debugging
        logger.info(f"Raw model output: {response}")
        
        # Process the model output
        paraphrased_text = postprocess_text(response)
        
        if paraphrased_text is None:
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
            "initialized": paraphraser is not None
        }
    }
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
