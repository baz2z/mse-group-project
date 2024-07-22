import React from 'react';
import { createColorMap, linearScale } from "@colormap/core";
import { cividis, viridis, magma, inferno, plasma, blackWhite } from "@colormap/presets";

const ColorBar = () => {
  const colors = magma;
  const domain = [0, 1];
  const range = [0, 1];
  const scale = linearScale(domain, range);
  const colorMap = createColorMap(colors, scale);

  // Generate gradient steps for the color map
  const gradientSteps = Array.from({ length: 100 }, (_, i) => {
    const [r, g, b] = colorMap(i / 100);
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  });

  // Update gradient direction to flow from left to right
  const gradient = `linear-gradient(to right, ${gradientSteps.join(', ')})`;

  return (
    <div className="flex flex-row items-center text-center">
      <span className="mr-2 text-sm font-semibold" style={{ color: '#0A2540' }}>
        Highly Relevant
      </span>
      <div
        className="w-32 h-6"  // Adjust width and height for horizontal bar
        style={{ background: gradient }}
      ></div>
      <span className="ml-2 text-sm font-semibold" style={{ color: '#0A2540' }} >
        Less Relevant
      </span>
    </div>
  );
};

export default ColorBar;
