import { useState } from 'react';
import '../app/search-result.css';
import { createColorMap, linearScale } from "@colormap/core";
import { viridis, plasma, inferno, magma, cividis } from "@colormap/presets";

const AdvancedSearchResult = ({ searchResult }) => {
  const [showPreview, setShowPreview] = useState(false);

  const handleMouseEnter = () => setShowPreview(true);
  const handleMouseLeave = () => setShowPreview(false);

  // Define your colormap using the viridis preset
  const colors = cividis  ; // Viridis colormap
  const domain = [0, 1]; // Assuming scores are between 0 and 1
  const range = [0, 1];
  const scale = linearScale(domain, range);
  const colorMap = createColorMap(colors, scale);

  // Get the background color based on the relevancy score
  const getBackgroundColor = (score) => {
    const [r, g, b] = colorMap(score);
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  };

  const itemClasses = 'p-4 border rounded-lg transition-transform';

  return (
    <div 
      className={`search-result-item min-w-96 ${itemClasses}`}
      style={{ backgroundColor: getBackgroundColor(searchResult.score) }}
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

export default AdvancedSearchResult;
