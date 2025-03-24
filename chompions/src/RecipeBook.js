import React from "react";
import "./RecipeBook.css";

function RecipeBook({ recipes, detectedIngredients }) {
  return (
    <div className="recipe-book">
      <div className="header-section">
        <h2 className="book-title">📖 Recipe Book</h2>
        {detectedIngredients.length > 0 && (
          <div className="detected-section">
            <h3 className="detected-title">Detected Ingredients:</h3>
            <div className="ingredient-pills">
              {detectedIngredients.map((ingredient, index) => (
                <span key={index} className="ingredient-pill">
                  {ingredient}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="recipes-grid">
        {recipes.length > 0 ? (
          recipes.map((recipe, index) => (
            <div key={index} className="recipe-card">
              <div className="card-header">
                <h3 className="recipe-name">{recipe.name}</h3>
                <div className="recipe-meta">
                  {recipe.cooking_time && (
                    <span className="meta-item">
                      ⏱ {recipe.cooking_time} mins
                    </span>
                  )}
                  {recipe.difficulty && (
                    <span className="meta-item">🎚 {recipe.difficulty}</span>
                  )}
                  {recipe.servings && (
                    <span className="meta-item">
                      👥 Serves {recipe.servings}
                    </span>
                  )}
                </div>
              </div>

              <div className="card-content">
                <div className="ingredients-section">
                  <h4 className="section-title">Ingredients</h4>
                  <ul className="ingredients-list">
                    {recipe.ingredients.map((ingredient, i) => (
                      <li key={i} className="ingredient-item">
                        {ingredient}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="instructions-section">
                  <h4 className="section-title">Instructions</h4>
                  <ol className="instructions-list">
                    {recipe.instructions.split("\n").map((step, i) => (
                      <li key={i} className="instruction-step">
                        {step}
                      </li>
                    ))}
                  </ol>
                </div>

                {recipe.tags && (
                  <div className="tags-section">
                    {recipe.tags.map((tag, i) => (
                      <span key={i} className="recipe-tag">
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="no-recipes">
            <img
              src="/empty-state.svg"
              alt="No recipes found"
              className="empty-icon"
            />
            <p className="empty-text">
              No matching recipes found. Try uploading a different image!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default RecipeBook;
