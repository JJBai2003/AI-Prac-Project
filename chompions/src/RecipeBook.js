import React from "react";
import { useLocation, useNavigate } from "react-router-dom"; 
import "./RecipeBook.css";

function RecipeBook() {
  const location = useLocation();
  const navigate = useNavigate(); 

  const { generatedRecipe } = location.state || {};

  return (
    <div className="book-wrapper">
      <div>
        <h2>Recipe Book</h2>
      </div>
      <div className="book-container">
        <div className="recipes-section">
          <h3>Recipes:</h3>
          <ul>
            {generatedRecipe?.length > 0 ? (
              <pre className="generated-recipe">{generatedRecipe}</pre>
            ) : (
              <p>No recipes found.</p>
            )}
          </ul>
        </div>
        <button onClick={() => navigate("/")} className="go-back-button">
          Upload New Photos
        </button>
      </div>
    </div>
  );
}

export default RecipeBook;
