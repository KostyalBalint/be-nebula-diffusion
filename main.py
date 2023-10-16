import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv(".env")

app = Flask(__name__)

# Sample data as a list of items
items = [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"}
]


# Define a route to get all items
@app.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)


# Define a route to get a single item by ID
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"message": "Item not found"}), 404
    return jsonify(item)


# Define a route to create a new item
@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    if "name" not in data:
        return jsonify({"message": "Name is required"}), 400
    new_item = {
        "id": len(items) + 1,
        "name": data["name"]
    }
    items.append(new_item)
    return jsonify(new_item), 201


if __name__ == '__main__':
    port = os.getenv("PORT", 5000)
    host = os.getenv("HOST", "localhost")
    app.run(debug=True, port=port, host=host)
