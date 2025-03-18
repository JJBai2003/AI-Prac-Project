import './App.css';
import photoimg from './assets/photo.png';

function App() {
  return (
    <div>
      <div className="outer-container">
        <h1>Discover New Recipes</h1>
        <div className="container">
          <label htmlFor="file-upload" className="smaller-container">
            <img src={photoimg} alt="img of a photo icon" />
            <p><span>Click</span> to upload a photo of a refrigerator</p>
            <input type="file" id="file-upload" name="file" className="file-input" />
          </label>
        </div>
      </div>
    </div>
  );
}

export default App;
