from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a paraphrasing model
model_name = "t5-small"  # Replace with a better model if needed
paraphraser = pipeline("text2text-generation", model=model_name)

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.json
    if 'text' not in data:
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
    app.run(host='0.0.0.0', port=8080)
