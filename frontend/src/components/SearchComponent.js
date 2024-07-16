"use client"; // This is a client component 👈🏽

import { useState } from 'react';
import axios from 'axios';
import { FaSearch, FaSpinner } from 'react-icons/fa';

import '../app/page-module.css';
import SearchResult from './SearchResult';
import categories from './categories';

const SearchComponent = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [showCategories, setShowCategories] = useState(false);

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
  const handleCategoryChange = (newCategory) => {
      if (selectedCategory === newCategory) {
          setSelectedCategory('');
        } else {
        setSelectedCategory(newCategory);
        console.log('Category changed to:', newCategory);
      }
  }
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
<div>
      <div style={{backdropFilter: 'blur(8px)'}} className="bg-white bg-opacity-10 rounded-lg shadow-md p-4 w-full max-w-lg mb-4 flex flex-col">
        {/* Search Component */}
        <div className="flex w-full max-w-md rounded-full overflow-hidden bg-white" style={{ minWidth: '28rem' }}>
        <input
          type="text"
          placeholder="Enter your search query"
          className="flex-1 p-4 text-gray-700 outline-none min-w-96"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyUp={handleKeyPress} // Trigger handleKeyPress on key press
        />
        <button
          className="flex items-center justify-center px-4 bg-white text-gray-700 rounded-r-full shadow-md ml-1 border border-gray-300"
          onClick={handleSearch}
        >
          <FaSearch />
        </button>
      </div>
    {/*Switch */}
    <div className="flex items-center mt-4">
      <label className="flex items-center cursor-pointer relative">
        <input
          type="checkbox"
          className="hidden"
          checked={showCategories}
          onChange={() => setShowCategories(!showCategories)}
        />
        <div className="w-10 h-4 bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 rounded-full shadow-inner relative"></div>
        <div
          className={`dot w-6 h-6 rounded-full shadow absolute top-[-0.25rem] left-[-0.25rem] transition-transform ${
            showCategories ? 'transform translate-x-6 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500' : 'bg-white'
          }`}
        ></div>
      </label>
      <span 
        className={`ml-3 text-gray-500 cursor-pointer ${showCategories ? 'bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500' : ''}`}
        onClick={() => setShowCategories(!showCategories)}
      >
        {showCategories ? 'Advanced Search' : 'Advanced Search'}
      </span>
    </div>

        {/* Category Selection Buttons */}
        <div
          className="overflow-hidden transition-max-height duration-[2000ms]"
          style={{ maxHeight: showCategories ? '1000px' : '0' }}
        >
          {showCategories && (
            <div className="flex flex-wrap justify-center mt-2 max-w-md mx-auto">
              {Object.keys(categories).map((category) => (
                <button
                  key={category}
                  onClick={() => handleCategoryChange(category)}
                  className={`m-2 px-4 py-1.5 rounded-full ${
                    category === selectedCategory ? 'bg-gray-300 text-gray-800 border border-transparent' : 'bg-transparent text-white border border-white'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          )}
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
