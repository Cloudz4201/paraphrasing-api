import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model name and debug mode from environment variables
model_name = os.getenv("MODEL_NAME", "t5-small")  # Default to "t5-small"
debug_mode = os.getenv("DEBUG", "False").lower() == "true"

# Initialize the paraphrasing model
paraphraser = pipeline("text2text-generation", model=model_name)

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    original_text = data['text']
    try:
        # Generate paraphrased text
        response = paraphraser(f"paraphrase: {original_text}", max_length=200, num_return_sequences=1)
        paraphrased_text = response[0]['generated_text']
        return jsonify({"paraphrased_text": paraphrased_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
