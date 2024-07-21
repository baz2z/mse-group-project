"use client"; // This is a client component 👈🏽

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { FaSpinner, FaArrowLeft } from 'react-icons/fa';
import AdvancedSearchResult from '../../components/AdvancedSearchResult';
import ColorBar from '@/components/ColorBar';

export default function SearchResults() {
  const searchParams = useSearchParams();
  const query = searchParams.get('query');
  const [results, setResults] = useState({ top_results: [], min_score: 0, max_score: 1 });
  const [loading, setLoading] = useState(false);
  const [showAllResults, setShowAllResults] = useState(false);

  const router = useRouter();
  const handleToggle = () => {
    setShowAllResults((prevState) => !prevState);
  };

  useEffect(() => {
    if (query) {
      // Check if results are already in sessionStorage
      const cachedResults = sessionStorage.getItem(`search-results-${query}`);
      if (cachedResults) {
        console.log('Using cached results for query', query);
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
      setResults({ top_results: [], min_score: 0, max_score: 1 }); // Clear previous results

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
    <>
      {/* Background Overlay */}
      <div className="fixed inset-0 bg-gray-100 bg-opacity-30 z-0"></div>
      {/* Content */}
      <div className="relative flex min-h-screen flex-col items-center p-24 pt-28 z-10">
        <div className="flex flex-row items-start w-full max-w-4xl">
          <div className="flex flex-col items-center w-full max-w-4xl">
            <div className="flex flex-col items-baseline justify-around w-full max-w-4xl">
              <div className="flex flex-row items-baseline justify-between w-full max-w-4xl">
                <h1 className="text-lg font-bold" style={{ color: '#0A2540' }}>Search Results for: {query}</h1>
                <ColorBar />
              </div>
              <div className="flex flex-row items-baseline justify-between w-full max-w-4xl mt-4">
                <button
                  onClick={() => router.push('/')}
                  className="flex items-center border border-blue-500 text-blue-500 hover:text-white hover:bg-blue-500 transition-colors duration-300 rounded-full p-2"
                >
                  <FaArrowLeft className="mr-2 w-5 h-5" />
                  Back to home
                </button>
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={showAllResults}
                    onChange={handleToggle}
                    className="form-checkbox h-5 w-5 text-blue-600"
                  />
                  <span className="text-blue-500">Show All Results</span>
                </label>
              </div>
            </div>
            {loading && (
              <div className="flex flex-col items-center mt-4 text-gray-600">
                <FaSpinner className="animate-spin mr-2 w-8 h-8 text-blue-500" />
                <span className="text-gray-700 text-lg">Loading...</span>
              </div>
            )}
            {results.top_results.length > 0 && (
              <div className="flex flex-row items-center justify-center w-full">
                <ul className="mt-4 flex flex-col items-center w-full">
                  {results.top_results.slice(0, showAllResults ? results.top_results.length : 3).map((result, index) => (
                    <AdvancedSearchResult key={index} searchResult={result} min_score={results.min_score} max_score={results.max_score} />
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
