
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from routes.recipe_routes import recipe_bp
import os

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app, resources={
        r"/api/*": {
            "origins": "http://localhost:3000",
            "allow_headers": ["Content-Type", "Authorization"],
            "methods": ["GET", "POST", "PUT", "DELETE"]
        }
    })
    
    
    #  upload folder
    upload_folder = os.getenv("UPLOAD_FOLDER")
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    print("here is the " ,upload_folder)
    # Register blueprints
    app.register_blueprint(recipe_bp, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    from config.db import create_indexes, setup_schema
    try:
        create_indexes()
        setup_schema()
        app = create_app()
        port = int(os.getenv("FLASK_PORT", 1))  # Get port from .env
        print(f"🚀 Server running on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"🔥 Failed to start server: {e}")