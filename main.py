import os
import downloader

import objaverse
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from get_point_cloud_by_uid import get_point_cloud_by_uid

load_dotenv(".env")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


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


if __name__ == '__main__':
    port = os.getenv("PORT", 5000)
    host = os.getenv("HOST", "localhost")
    app.run(debug=True, port=port, host=host)
