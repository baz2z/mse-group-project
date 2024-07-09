// pages/api/search.js

export default async function handler(req, res) {
    const { query } = req.query;
  
    // Simulate a search operation (e.g., query a database or external API)
    const dummyResults = [
      {
        link: 'https://uni-tuebingen.de/',
        title: 'Example Page 1',
        abstract: 'This is a summary of example page 1.',
      },
      {
        link: 'https://www.tuebingen-info.de',
        title: 'Example Page 2',
        abstract: 'This is a summary of example page 2.',
      },
      // Add more results as needed
    ];
    
  
    res.status(200).json(dummyResults);
  }
  