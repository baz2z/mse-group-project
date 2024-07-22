"use client"; // This is a client component 👈🏽

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { FaSpinner, FaArrowLeft } from 'react-icons/fa';
import AdvancedSearchResult from '../../components/AdvancedSearchResult';
import ColorBar from '@/components/ColorBar';

export default function SearchResults() {
  const json_form = {
    'min_score': 0,
    'max_score': 1,
    'top_results': [],
  };
  const [results, setResults] = useState(json_form);
  const [loading, setLoading] = useState(false);
  const [showAllResults, setShowAllResults] = useState(false);
  const router = useRouter();
  
  const searchParams = useSearchParams();
  const query = searchParams.get('query');
  const category = searchParams.get('category');
  
  const initialRender = useRef(true);

  const handleToggle = () => {
    setShowAllResults((prevState) => !prevState);
  };

  useEffect(() => {
    if (initialRender.current) {
      initialRender.current = false;
      return;
    }

    if (query) {
      // Check if results are already in sessionStorage
      const cachedResults = sessionStorage.getItem(`search-results-${query}`);
      if (cachedResults) {
        console.log('Using cached results for query', query);
        setResults(JSON.parse(cachedResults));
      } else {
        // Debouncing the API call
        const timer = setTimeout(() => {
          fetchResults(query, category);
        }, 300);
        return () => clearTimeout(timer);
      }
    }
  }, [query, category]);

  const fetchResults = async (searchQuery, category) => {
    console.log('Searching for:', searchQuery, 'with category:', category);
    try {
      setLoading(true);
      setResults(json_form); // Clear previous results

      // Make the API request
      const response = await axios.get('http://localhost:8000/search', {
        params: {
          query: searchQuery,
          category: category,
        },
      });

      console.log("response", response);

      setResults(response.data); // Assuming the search results are returned in response.data
      sessionStorage.setItem(`search-results-${searchQuery}`, JSON.stringify({
        'min_score': response.data.min_score, 
        'max_score': response.data.max_score, 
        'top_results': response.data.top_results}
      ));
      setLoading(false); // Set loading to false after results are fetched
    } catch (error) {
      console.error('Error fetching search results:', error);
      setLoading(false); // Make sure to set loading to false in case of error
    }
  };

  return (
    <>
      {/* Background Overlay */}
      <div className="fixed inset-0 bg-gray-100 bg-opacity-10 z-0"></div>
      {/* Content */}
      <div className="relative flex min-h-screen flex-col items-center p-24 pt-12 z-10">
        <div className="flex flex-row items-start w-full max-w-4xl">
          <div className="flex flex-col items-center w-full max-w-4xl">
            <div className="flex flex-col items-center justify-around w-full max-w-4xl">
              <div className="bg-white bg-opacity-90 rounded-lg shadow-md p-6 w-full">
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
                  <div className="flex items-center">
                    <div className="relative flex border rounded-full overflow-hidden">
                      <input
                        type="radio"
                        id="showAll"
                        name="showResults"
                        checked={showAllResults}
                        onChange={() => handleToggle(true)}
                        className="hidden"
                      />
                      <input
                        type="radio"
                        id="hideResults"
                        name="showResults"
                        checked={!showAllResults}
                        onChange={() => handleToggle(false)}
                        className="hidden"
                      />
                      <label
                        htmlFor="showAll"
                        className={`px-6 py-2 flex items-center justify-center cursor-pointer transition-colors duration-300 ${showAllResults ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-200 text-blue-500 border-gray-300'}`}
                      >
                        Show All Results
                      </label>
                      <label
                        htmlFor="hideResults"
                        className={`px-6 py-2 flex items-center justify-center cursor-pointer transition-colors duration-300 ${!showAllResults ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-200 text-blue-500 border-gray-300'}`}
                      >
                        Hide Results
                      </label>
                    </div>
                  </div>
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
      </div>
    </>
  );
}
