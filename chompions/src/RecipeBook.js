import React from "react";
import { useLocation } from "react-router-dom";
import "./RecipeBook.css";

function RecipeBook() {
  const location = useLocation();

  console.log("location", location.state);
  const { generatedRecipe} = location.state || {};

  return (
    <div className="book-wrapper">
      <div>
        <h2>Recipe Book</h2>
      </div>
      <div className="book-container">
        <div className="recipes-section">
          <h3>Recipes:</h3>
          <ul>
            {generatedRecipe.length > 0 ? (
              <pre className="generated-recipe">{generatedRecipe}</pre>
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
