
from flask import Blueprint, request, jsonify
import os
import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from config.db import get_collection

load_dotenv()

recipe_bp = Blueprint('recipe', __name__, url_prefix='/api')
uploads_col = get_collection("uploads")
recipes_col = get_collection("recipes")

ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS").split(','))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@recipe_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Mock ingredients detection
            ingredients = ["eggs", "milk", "flour"]  # Replace with actual AI later
            
            # Find matching recipes
            query = {"ingredients": {"$all": ingredients}}
            matching_recipes = list(recipes_col.find(query, {'_id': 0}))
            
            # Store upload record
            upload_record = {
                "filename": filename,
                "ingredients": ingredients,
                "recipes_found": len(matching_recipes),
                "timestamp": datetime.datetime.utcnow()
            }
            uploads_col.insert_one(upload_record)
            
            return jsonify({
                "ingredients": ingredients,
                "recipes": matching_recipes
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({"error": "File type not allowed"}), 400