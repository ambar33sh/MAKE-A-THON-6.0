import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [category, setCategory] = useState('');
  const [price, setPrice] = useState('');
  const [rating, setRating] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict/', {
        category,
        price: parseFloat(price),
        rating: parseFloat(rating),
        other_features: {}
      });

      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  return (
    <div className="App" style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>E-Commerce Product Prediction</h1>
      
      <form onSubmit={handleSubmit}>
        <div>
          <label>Category:</label>
          <input 
            type="text" 
            value={category} 
            onChange={(e) => setCategory(e.target.value)} 
          />
        </div>
        <div>
          <label>Price:</label>
          <input 
            type="text" 
            value={price} 
            onChange={(e) => setPrice(e.target.value)} 
          />
        </div>
        <div>
          <label>Rating:</label>
          <input 
            type="text" 
            value={rating} 
            onChange={(e) => setRating(e.target.value)} 
          />
        </div>
        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <div>
          <h2>Prediction Result: {prediction}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
