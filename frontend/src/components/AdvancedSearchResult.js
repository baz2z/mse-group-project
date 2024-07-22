import { useState, useEffect, useRef } from 'react';
import '../app/search-result.css';
import { createColorMap, linearScale } from "@colormap/core";
import { magma } from "@colormap/presets";
import { FaExternalLinkAlt, FaClipboard, FaEye, FaCheck } from 'react-icons/fa';

const AdvancedSearchResult = ({ searchResult, min_score, max_score }) => {
  const [showPreview, setShowPreview] = useState(false);
  const [iframeSrc, setIframeSrc] = useState('');
  const [copied, setCopied] = useState(false);
  const [previewError, setPreviewError] = useState(false);
  const iframeRef = useRef(null);

  const reversedColors = [...magma].reverse(); // Reverse the colormap
  const domain = [min_score, max_score];
  const range = [0, 1];
  const scale = linearScale(domain, range);
  const colorMap = createColorMap(reversedColors, scale);

  const getBackgroundColor = (score) => {
    const [r, g, b] = colorMap(score);
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  };

  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(searchResult.url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000); // Reset after 2 seconds
  };

  const dist = JSON.parse(searchResult.dist);

  useEffect(() => {
    // Set the iframe source to preload the content
    const url = searchResult.url;
    setIframeSrc(url);
    setPreviewError(false);
  }, [searchResult.url]);

  return (
    <div className={`search-result-item bg-white shadow-md p-6 rounded-lg w-full max-w-xl mb-4`} style={{ position: 'relative' }}>
      <div className="flex flex-col mb-4 result-content">
        <div className="flex items-center">
          <a
            href={searchResult.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:underline cursor-pointer"
          >
            {searchResult.main_domain}
          </a>
        </div>
        <p className="text-gray-500 mb-3 text-sm">Last Access on {new Date(searchResult.created).toLocaleDateString('en-GB')}</p>
        <h3 className="text-lg font-bold mb-2 result-title">{searchResult.title || 'Unknown Title'}</h3>
        <p className="text-gray-600 text-sm mb-2 result-abstract">{searchResult.description || 'No description available.'}</p>
      </div>
      <div className="dist-bar flex justify-between mb-4 border border-gray-300 bg-gradient-to-r via-purple-500 to-pink-500 p-1 rounded-xl">
        {dist.map((sectionScore, index) => (
          <div
            key={index}
            className="flex-1 h-2 mr-1 last:mr-0"
            style={{ backgroundColor: getBackgroundColor(sectionScore) }}
          ></div>
        ))}
      </div>
      <div className="action-buttons flex justify-between mt-2 text-blue-500">
        <div className='flex'>
          <a
            href={searchResult.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition mr-3"
          >
            <FaExternalLinkAlt className="mr-1" /> Go to website
          </a>
          <button
            onClick={handleCopyToClipboard}
            className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition mr-3"
          >
            {copied ? (
              <>
                <FaCheck className="mr-1 text-blue-500" /> Copied!
              </>
            ) : (
              <>
                <FaClipboard className="mr-1" /> Copy URL
              </>
            )}
          </button>
        </div>
        <button
          onClick={() => setShowPreview(!showPreview)}
          className="flex items-center text-[#0A2540] hover:text-[#136ef8] transition"
        >
          <FaEye className="mr-1" /> {showPreview ? <s>Preview</s> : 'Preview'}
        </button>
      </div>
      <div className={`result-preview ${showPreview ? 'visible' : 'hidden'}`}>
        {previewError ? (
          <div className="w-full h-full flex items-center justify-center text-gray-600">
            Preview not available.
          </div>
        ) : (
          <iframe
            src={iframeSrc}
            title="preview"
            className="w-full h-full border-none"
            sandbox="allow-same-origin allow-scripts"
            ref={iframeRef}
            onLoad={() => setPreviewError(false)}
            onError={() => setPreviewError(true)}
          ></iframe>
        )}
      </div>
    </div>
  );
};

export default AdvancedSearchResult;
