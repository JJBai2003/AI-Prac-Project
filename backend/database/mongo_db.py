from pymongo import MongoClient

def init_db():
    client = MongoClient("mongodb://localhost:1234/")
    db = client["recipe_ai"]

    if "users" not in db.list_collection_names():
        db.create_collection("users")
    if "recipes" not in db.list_collection_names():
        db.create_collection("recipes")

    return db

def add_recipe(db, name, ingredients, dietary_preference):
    recipe = {
        "name": name,
        "ingredients": ingredients,
        "dietary_preference": dietary_preference
    }
    db.recipes.insert_one(recipe)

def get_recipes(db):
    return list(db.recipes.find({}, {"_id": 0}))
