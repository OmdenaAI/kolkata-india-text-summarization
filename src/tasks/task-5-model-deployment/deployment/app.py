from flask import Flask, request, jsonify ,render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer
model_name = "t5-base"  # Choose a T5 variant suitable for text summarization
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json  # Assuming JSON data
        input_text = data['input_text']

        # Tokenize and generate summary
        input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"summary": generated_summary})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
