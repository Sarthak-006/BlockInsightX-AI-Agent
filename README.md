# BlockInsight X - AI Agent

An AI-powered analytics platform for the MultiversX blockchain, bringing Satoshi's vision to blockchain analysis for the AI MegaWave Hackathon.

## Features

- Transaction Analysis with AI insights
- Market Analysis and Predictions
- Network Statistics Visualization
- Real-time data from MultiversX blockchain
- Rate limit optimized for Groq API
- **Satoshi's Dashboard** with cryptoeconomic analysis

## Satoshi-Inspired Analysis

This agent implements core blockchain principles pioneered by Satoshi Nakamoto:

- **Consensus Health Analysis**: Measure centralization risk with Nakamoto coefficient
- **Double-Spend Detection**: Analyze transaction patterns to identify potential double-spend attempts
- **Cryptographic Verification**: Validate transaction signatures and merkle proofs
- **Token Velocity Analysis**: Evaluate if the token functions as store-of-value or medium-of-exchange
- **Economic Incentive Modeling**: Assess staking equilibrium and utility-to-speculation ratio
- **Contract Security Auditing**: Identify potential vulnerabilities in smart contracts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Select analysis type from the sidebar
2. For Transaction Analysis: Enter a transaction hash
3. For Market Insights: View AI-generated market analysis and trends
4. For Network Stats: Monitor real-time network statistics
5. For Satoshi's Dashboard: See comprehensive blockchain health metrics

## Rate Limit Optimizations

This application includes several optimizations to handle Groq API rate limits:

- Uses smaller LLM model (llama3-8b-8192) with more efficient token usage
- Selectively extracts important data to reduce token consumption
- Implements time-based request throttling
- Handles rate limit errors gracefully with automatic retries
- Provides user feedback during rate limit delays
- Compresses prompts and responses to stay within the 5000 tokens/minute limit

## Technologies Used

- Streamlit for UI
- Groq for AI analysis with rate limit handling
- MultiversX SDK for blockchain interaction
- Plotly for data visualization
- Advanced cryptoeconomic models for blockchain health analysis

## Contributing

This project is part of the MultiversX AI MegaWave Hackathon. Feel free to submit issues and enhancement requests. 