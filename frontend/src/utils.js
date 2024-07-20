export async function getPageTitle(url) {
    try {
      const response = await fetch(url);
      const html = await response.text();
      
      // Create a temporary div element to parse the HTML
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = html;
      
      // Get the title element
      const titleElement = tempDiv.querySelector('title');
      
      // Extract the title text content
      if (titleElement) {
        return titleElement.textContent;
      } else {
        return 'Title not found';
      }
    } catch (error) {
      console.error('Error fetching or parsing the HTML:', error);
      return 'Error fetching or parsing the HTML';
    }
  }