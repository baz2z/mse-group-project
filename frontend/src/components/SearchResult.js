import { useState } from 'react';

import '../app/search-result.css'
// TODO: load html from link;  get title 
const SearchResult = ({ searchResult }) => {
  const [showPreview, setShowPreview] = useState(false);

  const handleMouseEnter = () => setShowPreview(true);
  const handleMouseLeave = () => setShowPreview(false);

  return (
    <div 
      className="search-result-item"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <a href={searchResult.url} target="_blank" rel="noopener noreferrer">
        <div className="result-content">
          <h3 className="result-title">{searchResult.title}</h3>
          <p className="result-abstract">{searchResult.score}</p>
        </div>
      </a>
      {showPreview && (
        <div 
          className="result-preview"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          {/* include sandboxing for security during a deployment: sandbox="allow-same-origin" */}
          <iframe src={searchResult.url} title="preview" ></iframe>
        </div>
      )}
    </div>
  );
}

export default SearchResult;
