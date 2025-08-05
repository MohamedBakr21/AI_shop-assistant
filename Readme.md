# AI Shopping Assistant (Flask + SerpAPI + Groq)

This is a **Flask web application** that:
- Refines user search queries using Groq's model (via LangChain).
- Retrieves real-time product data from Google Shopping using **SerpAPI**.
- Displays a comparison table and AI-generated product summary.

---

## Features
- **Query Refinement**: model improves and enriches search terms.
- **Location-Aware Search**: Detects ISO 3166-1 alpha-2 country codes for accurate results.
- **Real-Time Product Data**: Fetches shopping results from SerpAPI.
- **Comparison Table**: Clickable links, prices, ratings, reviews, and images.
- **AI-Powered Summary**: Highlights best value, top ratings, unique features, and trade-offs.
