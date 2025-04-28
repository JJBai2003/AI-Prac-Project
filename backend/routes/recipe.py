from fastapi import APIRouter, File, UploadFile
from utils.image_processing import convert_image_to_tensor
from utils.model_inference import get_ingredients_from_image
from utils.recipe_generator import generate_recipe
from database import store_recipe

router = APIRouter()

@router.post("/upload")
async def upload_food_image(image: UploadFile = File(...)):
    image_tensor = await convert_image_to_tensor(image)
    ingredients = get_ingredients_from_image(image_tensor)
    recipe = generate_recipe(ingredients)
    store_recipe(recipe)
    return {"ingredients": ingredients, "recipe": recipe}
