"use client"; // This is a client component 👈🏽

import { useState } from 'react';
import axios from 'axios';
import { FaSearch, FaSpinner } from 'react-icons/fa';

import '../app/page-module.css';
import SearchResult from './SearchResult';

const SearchComponent = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [category, setCategory] = useState('');

  const handleSearch = async () => {
    try {
      setLoading(true); // Set loading to true before making the request
      setResults([]); // Clear the previous search results (if any
      console.log('Searching for:', query);
      // nextjs api route call
      // const response = await axios.get(`/api/search?query=${query}`);
      // fast api route call
      const response = await axios.get(`http://localhost:8000/search`, {
        params: { query }
      });
      console.log("response", response);
      // Simulate 1 second delay with setTimeout
      setTimeout(() => {
        setResults(response.data.results); // Assuming the search results are returned in response.data
        setLoading(false); // Set loading to false after results are fetched
      }, 2000);
    } catch (error) {
      console.error('Error fetching search results:', error);
      setLoading(false); // Make sure to set loading to false in case of error
    }
  };
  const handleCategoryChange = (selectedCategory) => {
      if (category === selectedCategory) {
          setCategory('');
        } else {
        setCategory(selectedCategory);
        console.log('Category changed to:', selectedCategory);
      }
  }

  return (
<div>
    <div style={{backdropFilter: 'blur(8px)'}} className="bg-white bg-opacity-10 rounded-lg shadow-md p-4 w-full max-w-xl mb-4">
      {/* Search Component */}
      <div className="flex w-full max-w-xl rounded-full overflow-hidden bg-white">
        <input
          type="text"
          placeholder="Enter your search query"
          className="flex-1 p-4 text-gray-700 outline-none min-w-96"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
        className="flex items-center justify-center px-4 bg-white text-gray-700 rounded-r-full shadow-md ml-1 border border-gray-300"
        onClick={handleSearch}
            >
          <FaSearch />
        </button>
      </div>

      {/* Category Selection Buttons */}
      <div className="flex justify-center mt-2">
        <button
          className={`m-2 px-4 py-1.5 rounded-full ${category === 'category1' ? 'bg-gray-300 text-gray-800' : 'bg-transparent text-white border border-white'}`}
          onClick={() => handleCategoryChange('category1')}
        >
          Restaurants
        </button>
        <button
          className={`m-2 px-4 py-1.5 rounded-full ${category === 'category2' ? 'bg-gray-300 text-gray-800' : 'bg-transparent text-white border border-white'}`}
          onClick={() => handleCategoryChange('category2')}
        >
          Attractions
        </button>
        {/* Add more buttons for additional categories as needed */}
      </div>
    </div>

  {/* Loading Indicator */}
  {loading && (
    <div className="flex flex-col items-center mt-4 text-gray-600">
      <FaSpinner className="animate-spin mr-2 w-8 h-8 text-white" />
      <span className="text-white">Loading...</span>
    </div>
  )}

  {/* Search Results */}
  {results.length > 0 && (
    <ul className="mt-4 flex flex-col items-center">
      {results.map((result, index) => (
        <SearchResult key={index} searchResult={result}></SearchResult>
      ))}
    </ul>
  )}
</div>
  );
};

export default SearchComponent;
