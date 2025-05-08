import React from "react";
import { useLocation } from "react-router-dom";
import "./RecipeBook.css";

function RecipeBook() {
  const location = useLocation();
  const { recipes = [], detectedIngredients = [] } = location.state || {};

  return (
    <div className="book-wrapper">
      <div>
        <h2>Recipe Book</h2>
      </div>
      <div className="book-container">
        {/* Detected Ingredients Section */}
        <div className="ingredients-section">
          <h3>Detected Ingredients:</h3>
          <ul>
            {detectedIngredients.length > 0 ? (
              detectedIngredients.map((ingredient, index) => (
                <li key={index}>{ingredient}</li>
              ))
            ) : (
              <p>No ingredients detected.</p>
            )}
          </ul>
        </div>

        {/* Recipes Section */}
        <div className="recipes-section">
          <h3>Recipes:</h3>
          <ul>
            {recipes.length > 0 ? (
              recipes.map((recipe, index) => (
                <li key={index}>
                  <strong>{recipe.title}</strong>: {recipe.ingredients.join(", ")}
                </li>
              ))
            ) : (
              <p>No recipes found.</p>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default RecipeBook;