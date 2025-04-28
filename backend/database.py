from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = client["food_db"]
recipes_collection = db["recipes"]

def store_recipe(recipe: dict):
    recipes_collection.insert_one(recipe)
