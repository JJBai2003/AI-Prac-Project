import './App.css';
import NavBar from './components/NavBar';
import photo from './assets/photo.png'

function App() {
  return (
    <div>
      <NavBar></NavBar>
      <div className="outer-container">
        <h1>Discover New Recipes</h1>
        <div className="container">
          <div className="smaller-container">
            <img src={photo} alt="photo" />
            <p>Drop an image of your refrigerator here or <span>browse</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
