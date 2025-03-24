import React, { useState } from "react";
import "./App.css";
import photoimg from "./assets/photo.png";
import RecipeBook from "./RecipeBook";

function App() {
  const [recipes, setRecipes] = useState([]);
  const [ingredients, setIngredients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:5001/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server response was not OK");
      }

      const data = await response.json();
      setRecipes(data.recipes);
      setIngredients(data.ingredients);
    } catch (err) {
      setError("Failed to process image. Please try again.");
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="outer-container">
        <h1>Discover New Recipes</h1>
        <div className="container">
          <label htmlFor="file-upload" className="smaller-container">
            <img src={photoimg} alt="upload icon" />
            <p>
              <span>Click</span> to upload a photo of a refrigerator
            </p>
            <input
              type="file"
              id="file-upload"
              name="file"
              className="file-input"
              onChange={handleFileUpload}
              disabled={loading}
            />
          </label>
        </div>

        {loading && <div className="loading-spinner">Processing...</div>}
        {error && <div className="error-message">{error}</div>}
      </div>
      <RecipeBook recipes={recipes} detectedIngredients={ingredients} />
    </div>
  );
}

export default App;
