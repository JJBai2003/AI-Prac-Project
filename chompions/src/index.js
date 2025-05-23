import React from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";
import ReactDOM from 'react-dom/client';
import './index.css';
import NavBar from './components/NavBar';
import App from './App';
import Faq from './Faq';
import RecipeBook from './RecipeBook'
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(

  <React.StrictMode>
    <Router>
      <NavBar />
      <Routes>
        <Route exact path="/" element={<App />} />
        <Route path="/faq" element={<Faq />} />
        <Route path="/recipebook" element={<RecipeBook />} />

      </Routes>
    </Router>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
