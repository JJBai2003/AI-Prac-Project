import os
import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient, TEXT, DESCENDING
from pymongo.errors import OperationFailure

load_dotenv()

def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        db_name = os.getenv("DB_NAME")
        
       
        if db_name not in client.list_database_names():
            print(f"🆕 Creating database '{db_name}'")
            
        db = client[db_name]
        
     
        collections_to_create = ['recipes', 'uploads']
        for coll_name in collections_to_create:
            if coll_name not in db.list_collection_names():
                print(f"Creating collection '{coll_name}'")
                db.create_collection(coll_name)
        
        return db
        
    except Exception as e:
        print("Failed to connect to MongoDB:", e)
        raise

def get_collection(collection_name):
    return get_db()[collection_name]

def create_indexes():
    db = get_db()
    try:
        #  text index for recipe search
        db.recipes.create_index([("ingredients", TEXT)])
        # time-based index for uploads
        db.uploads.create_index([("timestamp", DESCENDING)])
        print("Database indexes created successfully")
    except OperationFailure as e:
        print("Failed to create indexes:", e)

def setup_schema():
    db = get_db()
    try:
        db.command({
            'collMod': 'recipes',
            'validator': {
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['name', 'ingredients', 'instructions'],
                    'properties': {
                        'name': {'bsonType': 'string'},
                        'ingredients': {
                            'bsonType': 'array',
                            'items': {'bsonType': 'string'}
                        },
                        'instructions': {'bsonType': 'string'},
                        'cooking_time': {'bsonType': 'int'},
                        'difficulty': {
                            'bsonType': 'string',
                            'enum': ['easy', 'medium', 'hard']
                        }
                    }
                }
            }
        })
        print("Database schema validation applied successfully")
    except OperationFailure as e:
        print("Failed to apply schema validation:", e)
       