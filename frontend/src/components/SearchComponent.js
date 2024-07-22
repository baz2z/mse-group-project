"use client"; // This is a client component 👈🏽

import { useRouter } from 'next/navigation'
import { useState } from 'react';
import { FaSearch, FaSpinner } from 'react-icons/fa';

import '../app/page-module.css';

import categories from '../constants';


const SearchComponent = () => {
  const [query, setQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [showCategories, setShowCategories] = useState(false);
  const router = useRouter();

  const handleSearch = () => {
    if (query.trim()) {
      const params = new URLSearchParams({ query: query });
      if (selectedCategory) {
        params.append('category', selectedCategory); // Or use the actual category if needed
      }
      router.push(`/search-results?${params.toString()}`);
    }
  };

  const handleCategoryChange = (newCategory) => {
    if (selectedCategory === newCategory) {
      setSelectedCategory('');
    } else {
      setSelectedCategory(newCategory);
    }
    
  }
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };


  return (
<div>
      <div style={{backdropFilter: 'blur(10px)'}} className="bg-white bg-opacity-10 rounded-lg shadow-md p-4 w-full max-w-lg mb-4 flex flex-col">
        {/* Search Component */}
        <div className="flex w-full max-w-md rounded-full overflow-hidden bg-white" style={{ minWidth: '30rem' }}>
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
        <div
          className={`w-10 h-4 rounded-full shadow-inner relative transition-all duration-[500ms] ease-in-out ${
            selectedCategory !== '' 
              ? 'bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300' 
              : 'bg-gray-200'
          }`}
        ></div>
        <div
          className={`dot w-6 h-6 rounded-full shadow absolute top-[-0.25rem] left-[-0.25rem] transition-transform ${
            showCategories 
              ? 'transform translate-x-6' 
              : 'transform translate-x-0'
          } ${
            selectedCategory !== '' 
              ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500' 
              : 'bg-gray-400'
          }`}
        ></div>
      </label>
      <span 
        className={`ml-3 cursor-pointer font-bold transition-all duration-[500ms] ease-in-out ${selectedCategory !== '' ? 'bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500' : 'text-gray-500'}`}
        onClick={() => setShowCategories(!showCategories)}
      >
        Advanced Search
      </span>
    </div>
{/* Category Selection Buttons */}
<div
  className={`overflow-hidden transition-all duration-[1000ms] ${showCategories ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}
>
  <div className="flex flex-wrap justify-center mt-2 max-w-md mx-auto">
    {Object.keys(categories).map((category) => (
      <button
        key={category}
        onClick={() => handleCategoryChange(category)}
        className={`m-2 px-4 py-1.5 rounded-full transition-colors duration-300 ${
          category === selectedCategory
            ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white shadow-inner'
            : 'text-white border border-white hover:bg-white hover:bg-opacity-50'
        }`}
      >
        {category}
      </button>
    ))}
      </div>
    </div>
  </div>
</div>
  );
};

export default SearchComponent;
