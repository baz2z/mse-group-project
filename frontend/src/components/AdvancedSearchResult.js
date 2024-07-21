import { useState } from 'react';
import '../app/search-result.css';
import { createColorMap, linearScale } from "@colormap/core";
import { viridis, plasma, inferno, magma, cividis } from "@colormap/presets";
import { FaExternalLinkAlt, FaClipboard, FaEye } from 'react-icons/fa';

const AdvancedSearchResult = ({ searchResult, min_score, max_score }) => {
  const [showPreview, setShowPreview] = useState(false);

  const [iframeAllowed, setIframeAllowed] = useState(true);

  const handleIframeLoad = (e) => {
    if (e.target.contentWindow.location !== searchResult.url) {
      setIframeAllowed(false);
    }
  };
  // Define your colormap using the viridis preset
  const colors = magma  ; // Viridis colormap
  const domain = [min_score, max_score]; // actual score range
  const range = [0, 1]; // colormap range
  const scale = linearScale(domain, range);
  const colorMap = createColorMap(colors, scale);

  // Get the background color based on the relevancy score
  const getBackgroundColor = (score) => {
    const [r, g, b] = colorMap(score);
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  };

  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(searchResult.url);
    // alert('URL copied to clipboard!');
  };

  const itemClasses = 'p-4 rounded-lg transition-transform transform hover:scale-105';
  // console.log("before searchResult", searchResult);
  let dist = JSON.parse(searchResult.dist);

  return (
<div 
  className={`search-result-item min-w-96 ${itemClasses} bg-white shadow-md p-4 rounded-lg`}
>
  <a href={searchResult.url} target="_blank" rel="noopener noreferrer">
    <div className="result-content mb-4">
      <h3 className="result-title text-lg font-bold">Titel Placeholder{searchResult.title}</h3>
      <p className="text-sm text-gray-600">Scraped on {new Date(searchResult.created).toLocaleDateString('en-GB')}</p>
      <p className="result-abstract">Max. Score: {searchResult.score.toFixed(2)}</p>
    </div>
  </a>
  <div className="dist-bar flex justify-between mt-2 border border-gray-300 bg-gradient-to-r via-purple-500 to-pink-500 p-1 rounded-xl">
    {dist.map((sectionScore, index) => (
      <div 
        key={index} 
        className="flex-1 h-2 mr-1 last:mr-0" 
        style={{ backgroundColor: getBackgroundColor(sectionScore) }}
      ></div>
    ))}
  </div>
  <div className="action-buttons flex justify-around mt-2 text-blue-500">
    <a href={searchResult.url} target="_blank" rel="noopener noreferrer" className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition">
      <FaExternalLinkAlt className="mr-1 icon text-[#0A2540]" /> Go to website
    </a>
    <button 
      onClick={handleCopyToClipboard} 
      className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition"
    >
      <FaClipboard className="mr-1 icon text-[#0A2540]" /> Copy URL
    </button>
    <button 
      onClick={() => setShowPreview(!showPreview)} 
      className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition"
    >
      <FaEye className="mr-1 icon text-[#0A2540]" /> <span onClick={() => setShowPreview(!showPreview)}>{showPreview ? <s>Preview</s> : 'Preview'}</span>
    </button>
  </div>
      <div 
        className={`result-preview mt-2 ${showPreview ? 'block' : 'hidden'}`}
      >
        {iframeAllowed ? (
          <iframe 
            src={searchResult.url} 
            title="preview" 
            className="w-full h-64 border-none"
            onLoad={handleIframeLoad}
            onError={() => setIframeAllowed(false)}
          ></iframe>
        ) : (
          <div className="w-full h-64 flex items-center justify-center text-gray-600">
            Preview not available.
          </div>
        )}
      </div>
</div>
  );
}

export default AdvancedSearchResult;
