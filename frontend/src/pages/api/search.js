// pages/api/search.js

export default async function handler(req, res) {
    const { query } = req.query;
  
    // Simulate a search operation (e.g., query a database or external API)
    const dummyResults = [
      `Result for "${query}" 1`,
      `Result for "${query}" 2`,
      `Result for "${query}" 3`,
    ];
  
    res.status(200).json(dummyResults);
  }
  