import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";
import photoimg from "./assets/photo.png";

function App() {
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  useEffect(() => {
    setPhotos([]);
    setError("");
  }, []);

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    const validPhotos = files.filter((file) => file.type.startsWith("image/"));
    if (validPhotos.length < files.length) {
      setError("Some files were not valid images and were skipped.");
    }
    setPhotos((prevPhotos) => [...prevPhotos, ...validPhotos]);
  };

  useEffect(() => {
    return () => {
      photos.forEach((photo) => URL.revokeObjectURL(photo));
    };
  }, [photos]);

  const handleSubmit = async () => {
    if (photos.length === 0) {
      setError("Please upload at least one photo.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      photos.forEach((photo) => {
        formData.append("file", photo);
      });

      const response = await fetch("http://localhost:5001/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server response was not OK");
      }

      const data = await response.json();
      console.log("Raw backend response:", data);

      const { generated_recipe, ingredients } = data;

      let ingredientsPerPhoto = [];
      if (Array.isArray(ingredients) && Array.isArray(ingredients[0])) {
        ingredientsPerPhoto = ingredients.map((ingredients, index) => ({
          photoName: photos[index]?.name || `Photo ${index + 1}`,
          ingredients: Array.isArray(ingredients) ? ingredients : [],
        }));
      } else if (Array.isArray(ingredients)) {
        ingredientsPerPhoto = [
          {
            photoName: "All Photos",
            ingredients: ingredients,
          },
        ];
      }

      const allIngredients = ingredientsPerPhoto.flatMap(
        (item) => item.ingredients
      );

      if (allIngredients.length === 0) {
        setError(
          "No ingredients detected. Please try uploading clearer photos."
        );
        return;
      }
      console.log("this is ", generated_recipe);

      navigate("/recipebook", {
        state: {
          recipes: data.recipes || [],
          generatedRecipe: generated_recipe,
          detectedIngredients: allIngredients,
          ingredientsPerPhoto,
        },
      });
    } catch (err) {
      setError("Failed to process images. Please try again.");
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
              <span>Click</span> to upload photos of your refrigerator
            </p>
            <input
              type="file"
              id="file-upload"
              name="file"
              className="file-input"
              onChange={handleFileUpload}
              multiple
            />
          </label>
        </div>
        <div className="uploaded-photos">
          {photos.map((photo, index) => (
            <div key={index} className="photo-preview">
              <img
                src={URL.createObjectURL(photo)}
                alt={`Uploaded ${index + 1}`}
              />
            </div>
          ))}
        </div>
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="submit-button"
        >
          {loading ? "Processing..." : "Submit Photos"}
        </button>

        {error && <div className="error-message">{error}</div>}
      </div>
    </div>
  );
}

export default App;
