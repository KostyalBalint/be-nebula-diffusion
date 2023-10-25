import os

from dotenv import load_dotenv
from flask import Flask, jsonify, Response
from flask_cors import CORS, cross_origin

import objaverse
from get_point_cloud_by_uid import get_point_cloud_by_uid
from generate_diffusion import generate_with_diffusion
from search import search_algolia

load_dotenv(".env")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/search/<string:query>', methods=['GET'])
@cross_origin()
def search_with_algolia(query: str):
    return jsonify(search_algolia(query))


# Define a route to get a single item by ID
@app.route('/annotation/<string:item_id>', methods=['GET'])
@cross_origin()
def get_item(item_id):
    try:
        annotation = objaverse.load_annotations([item_id])
        if annotation is None:
            return jsonify({"message": "Item not found"}), 404
        return jsonify(annotation)
    except:
        return jsonify({"message": "Item not found"}), 404


@app.route('/pointCloud/<string:uid>', methods=['GET'])
@cross_origin()
def get_point_cloud(uid):
    pc = get_point_cloud_by_uid(uid, scale=10)
    return jsonify(pc)


@app.route('/diffusionPointCloud/generate', methods=['GET'])
@cross_origin()
def gen_with_diffusion_point_cloud_stream():

    return Response(generate_with_diffusion(), mimetype='text/event-stream')


if __name__ == '__main__':
    port = os.getenv("PORT", 5000)
    host = os.getenv("HOST", "localhost")
    app.run(debug=True, port=port, host=host)
