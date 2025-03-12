from flask import Flask, request, jsonify
from flask_cors import CORS
from database.mongo_db import init_db, add_recipe, get_recipes

app = Flask(__name__)
CORS(app)

db = init_db()

@app.route('/add_recipe', methods=['POST'])
def add_recipe_endpoint():
    data = request.get_json()
    add_recipe(db, data['name'], data['ingredients'], data['dietary_preference'])
    return jsonify({"message": "Recipe added successfully"})

@app.route('/get_recipes', methods=['GET'])
def get_recipes_endpoint():
    recipes = get_recipes(db)
    return jsonify({"recipes": recipes})

if __name__ == '__main__':
    app.run(debug=True)
