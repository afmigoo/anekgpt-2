from flask import (
    Flask,
    request, abort,
    jsonify,
    render_template
)
from src.anek_gpt import generate
from src.anek_gpt.config import begin_tkn
from src.anek_gpt import static_model

static_model.load()

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/gen')
def gen():
    prompt = request.args.get('prompt')
    if not prompt:
        prompt = begin_tkn
    
    result = generate.generate(model=static_model.model, prompt=prompt)

    return jsonify({'result': result})
