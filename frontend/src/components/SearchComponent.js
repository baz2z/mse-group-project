"use client"; // This is a client component 👈🏽

import { useState } from 'react';
import axios from 'axios';
import { FaSearch } from 'react-icons/fa';

const SearchComponent = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    try {
      console.log('Searching for:', query);
      const response = await axios.get(`/api/search?query=${query}`);
      setResults(response.data); // Assuming the search results are returned in response.data
    } catch (error) {
      console.error('Error fetching search results:', error);
    }
  };

  return (
    <div className="mt-8 flex flex-col items-center justify-center">
      <div className="flex w-full max-w-xl rounded-full overflow-hidden border border-gray-300 bg-white">
        <input
          type="text"
          placeholder="Enter your search query"
          className="flex-1 p-4 text-gray-700 outline-none"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="flex items-center justify-center px-4 bg-blue-500 text-white rounded-r-full"
          onClick={handleSearch}
        >
          <FaSearch />
        </button>
      </div>
      <div className="mt-4 w-full max-w-xl">
        {results.length > 0 && (
          <ul className="mt-4">
            {results.map((result, index) => (
              <li key={index}>{result}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default SearchComponent;
