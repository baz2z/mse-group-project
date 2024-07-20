"use client"; // This is a client component 👈🏽

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { FaSpinner, FaArrowLeft } from 'react-icons/fa';

// import "../../app/globals.css";
import AdvancedSearchResult from '../../components/AdvancedSearchResult';

export default function SearchResults() {
  const searchParams = useSearchParams();
  const query = searchParams.get('query');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    if (query) {
      // Check if results are already in sessionStorage
      const cachedResults = sessionStorage.getItem(`search-results-${query}`);
      if (cachedResults) {
        console.log('Using cached results:', cachedResults)
        setResults(JSON.parse(cachedResults));
      } else {
        fetchResults(query);
      }
    }
  }, [query]);

  const fetchResults = async (searchQuery) => {
    console.log('Searching for:', searchQuery);
    try {
      setLoading(true);
      setResults([]); // Clear previous results

      // Make the API request
      const response = await axios.get('http://localhost:8000/search', {
        params: { query: searchQuery }
      });
      console.log("response", response);

      setTimeout(() => {
        setResults(response.data); // Assuming the search results are returned in response.data
        sessionStorage.setItem(`search-results-${searchQuery}`, JSON.stringify(response.data));
        sessionStorage.removeItem('search-query'); // Clear the query after search
        setLoading(false); // Set loading to false after results are fetched
      }, 1000);
    } catch (error) {
      console.error('Error fetching search results:', error);
      setLoading(false); // Make sure to set loading to false in case of error
    }
  };

  return (
    <div className="relative flex min-h-screen flex-col items-center p-24 pt-28 bg-gray-100">
      {/* Back to Home Button */}
      <button
        onClick={() => router.push('/')}
        className="absolute top-4 left-4 p-2 bg-blue-500 text-white rounded-full shadow-md hover:bg-blue-600 transition-transform transform hover:scale-105 active:bg-blue-700"
      >
        <FaArrowLeft className="w-5 h-5" />
      </button>

      <div className="flex flex-col items-center w-full max-w-4xl">
          <div className="flex flex-col items-start w-full max-w-4xl">
            <h1 className="text-lg font-bold mb-8">Search Results for: {query}</h1>
          </div>
            {loading && (
              <div className="flex flex-col items-center mt-4 text-gray-600">
                <FaSpinner className="animate-spin mr-2 w-8 h-8 text-blue-500" />
                <span className="text-gray-700 text-lg">Loading...</span>
              </div>
            )}
        {results.length > 0 && (
          <ul className="mt-4 flex flex-col items-center w-full">
            {results.map((result, index) => (
              <AdvancedSearchResult key={index} searchResult={result} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
