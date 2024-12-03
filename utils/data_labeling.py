# utils/data_labeling.py

from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
import base64
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
es = Elasticsearch()

@app.route('/', methods=['GET', 'POST'])
def label_data():
    if request.method == 'POST':
        doc_id = request.form['doc_id']
        new_label = request.form['label']
        es.update(index='mnist', id=doc_id, body={'doc': {'label': int(new_label)}})
        return 'Label updated!'
    else:
        res = es.search(index='mnist', body={"query": {"match_all": {}}}, size=1)
        if not res['hits']['hits']:
            return 'No data available for labeling.'
        doc = res['hits']['hits'][0]
        doc_id = doc['_id']
        source = doc['_source']
        # Convert pixel list back to image
        pixels = np.array(source['image']).reshape(28, 28)
        # Normalize to [0,255]
        pixels = (pixels * 255).astype(np.uint8)
        # Encode as base64
        img = Image.fromarray(pixels, 'L')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        label = source['label']
        return render_template('labeling.html', image=image_base64, label=label, doc_id=doc_id)

if __name__ == '__main__':
    app.run(debug=True)
