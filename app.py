from flask import Flask, request, render_template
from tool.predictor import Predictor

app = Flask(__name__)
predictor = Predictor(device='cpu', model_type='seq2seq', weight_path='./weights/seq2seq_0.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = predictor.predict(text.strip(), NGRAM=4)
        return render_template('index.html', original_text=text, corrected_text=corrected_text)

if __name__ == '__main__':
    app.run(debug=True)
