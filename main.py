import streamlit as st
import groq
import os
import pandas as pd
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import json
import time
import hashlib
import numpy as np
from collections import defaultdict

# Load environment variables
try:
    # First try from Streamlit secrets
    api_key = st.secrets["general"]["api_key"]
except Exception as e:
    # Fallback to environment variable if needed
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Error: Groq API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

# Initialize Groq client with the API key
client = Groq(api_key=api_key)

# MultiversX API endpoints
MULTIVERSX_API = "https://api.multiversx.com"
NETWORK = "mainnet"

# Rate limit constants
MAX_TOKENS_PER_MIN = 5000
TOKEN_RESET_INTERVAL = 60  # seconds

# Economic constants
VELOCITY_DAMPENING = 0.85  # Money velocity dampening factor
NETWORK_EFFECT_MULTIPLIER = 1.8  # Metcalfe's law parameter

# Streamlit configuration
st.set_page_config(
    page_title="BlockInsight X",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f8fa;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .satoshi-box {
        background-color: #f0f0f0;
        border-left: 5px solid #e6b800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .header-container {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .header-logo {
        font-size: 2rem;
        margin-right: 10px;
        color: #1f77b4;
    }
    .header-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #666;
        margin-left: 5px;
    }
    .card-container {
        background-color: white;
        border-radius: 5px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-top: 2rem;
    }
    .sidebar-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #d9534f;
        font-weight: 600;
    }
    .risk-medium {
        color: #f0ad4e;
        font-weight: 600;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: 600;
    }
    .tab-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #1f77b4;
    }
    .blockinsight-brand {
        font-weight: 700;
        color: #1f77b4;
    }
    .blockinsight-x {
        color: #333;
        font-weight: 900;
    }
    </style>
""", unsafe_allow_html=True)

class SatoshiConsensusAnalyzer:
    """Analyzes network consensus and potential forks"""
    
    def __init__(self):
        self.fork_threshold = 3  # Number of blocks to detect potential fork
        
    def detect_network_forks(self, recent_blocks):
        """Detects potential forks by analyzing block producers and timestamps"""
        # Check if recent_blocks is None, empty, or not a list/sequence
        if not recent_blocks or not hasattr(recent_blocks, '__len__') or len(recent_blocks) < 10:
            return {"status": "Not enough data", "fork_detected": False, 
                    "fork_risk": "Unknown", "nakamoto_coefficient": 0, 
                    "timing_anomalies": [], "centralization_risk": "Unknown"}
                    
        # Analyze block hash sequences and timing
        block_producers = [block.get("proposer", "") for block in recent_blocks]
        timestamps = [block.get("timestamp", 0) for block in recent_blocks]
        
        # Look for anomalies in block timing (potential fork indicator)
        time_diffs = [timestamps[i] - timestamps[i+1] for i in range(len(timestamps)-1)]
        mean_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        anomalies = [i for i, diff in enumerate(time_diffs) if diff > mean_time * 2]
        
        # Analyze distribution of validators (centralization risk)
        producer_counts = defaultdict(int)
        for producer in block_producers:
            producer_counts[producer] += 1
            
        # Calculate Nakamoto coefficient (decentralization metric)
        total_blocks = len(block_producers)
        sorted_producers = sorted(producer_counts.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0
        nakamoto_coef = 0
        
        for producer, count in sorted_producers:
            cumulative += count
            nakamoto_coef += 1
            if cumulative > total_blocks / 2:
                break
                
        fork_risk = "Low"
        if len(anomalies) >= self.fork_threshold:
            fork_risk = "High"
        elif len(anomalies) > 0:
            fork_risk = "Medium"
            
        return {
            "fork_risk": fork_risk,
            "nakamoto_coefficient": nakamoto_coef,
            "timing_anomalies": anomalies,
            "top_validators": sorted_producers[:5],
            "centralization_risk": "High" if nakamoto_coef <= 3 else "Medium" if nakamoto_coef <= 7 else "Low"
        }
        
class SatoshiEconomicModeler:
    """Implements economic models for token analysis"""
    
    def __init__(self):
        self.velocity_factor = VELOCITY_DAMPENING
        self.network_multiplier = NETWORK_EFFECT_MULTIPLIER
        
    def calculate_token_velocity(self, tx_volume, token_supply):
        """Calculate token velocity - a key indicator of token utility vs speculation"""
        if not token_supply:
            return 0
        velocity = tx_volume / token_supply
        return velocity
        
    def estimate_network_value(self, active_addresses, tx_volume):
        """Estimates network value based on Metcalfe's law and transaction volume"""
        # Metcalfe's law: value ~ n²
        metcalfe_value = self.network_multiplier * (active_addresses ** 2)
        
        # Adjust by transaction utility
        adjusted_value = metcalfe_value * (1 + np.log1p(tx_volume/1000))
        
        return adjusted_value
        
    def analyze_token_economics(self, market_data):
        """Performs advanced economic analysis of token metrics"""
        if not market_data:
            return {}
            
        # Extract key metrics
        active_addresses = market_data.get("accounts", 0)
        tx_volume = market_data.get("transactions", 0)
        token_supply = market_data.get("supply", {}).get("circulating", 0)
        
        # Calculate metrics
        token_velocity = self.calculate_token_velocity(tx_volume, token_supply)
        network_value = self.estimate_network_value(active_addresses, tx_volume)
        
        # Calculate token utility vs speculation ratio
        utility_ratio = min(1.0, (1.0/token_velocity) if token_velocity > 0 else 0)
        
        # Model staking economics
        staking_apy = market_data.get("staking", {}).get("apr", 0)
        staking_ratio = market_data.get("staking", {}).get("totalStaked", 0) / token_supply if token_supply else 0
        
        # Calculate staking equilibrium point
        equilibrium_staking = 0.5 * (1 - utility_ratio) + 0.3 * utility_ratio
        
        return {
            "token_velocity": token_velocity,
            "network_value": network_value,
            "utility_ratio": utility_ratio,
            "staking_ratio": staking_ratio,
            "staking_apy": staking_apy,
            "equilibrium_staking": equilibrium_staking,
            "staking_pressure": "Increasing" if staking_ratio < equilibrium_staking else "Decreasing",
            "economic_analysis": {
                "value_capture": "High" if utility_ratio > 0.7 else "Medium" if utility_ratio > 0.4 else "Low",
                "token_sink_effect": "Strong" if staking_ratio > 0.6 else "Moderate" if staking_ratio > 0.3 else "Weak",
                "velocity_sink": "Effective" if token_velocity < 4 else "Moderate" if token_velocity < 8 else "Ineffective"
            }
        }
        
class CryptographicVerifier:
    """Implements cryptographic verification of blockchain data"""
    
    def __init__(self):
        pass
        
    def verify_merkle_proof(self, tx_hash, merkle_proof, merkle_root):
        """Verify transaction inclusion with Merkle proof"""
        if not (tx_hash and merkle_proof and merkle_root):
            return False
            
        current = tx_hash
        for proof in merkle_proof:
            if proof.get("position") == "left":
                current = hashlib.sha256((proof.get("hash") + current).encode()).hexdigest()
            else:
                current = hashlib.sha256((current + proof.get("hash")).encode()).hexdigest()
                
        return current == merkle_root
        
    def verify_transaction_signature(self, tx_data):
        """Verify digital signature of transaction (conceptual)"""
        # In a real implementation, this would use specific MultiversX signature scheme
        # Here we're returning a dummy value since we don't have signature data
        return tx_data.get("signature") is not None
        
    def audit_contract_security(self, contract_data):
        """Analyze smart contract for common security vulnerabilities"""
        vulnerabilities = []
        
        # Check for reentrancy risk
        if "balance" in str(contract_data) and "transfer" in str(contract_data):
            vulnerabilities.append({
                "type": "Reentrancy Risk",
                "severity": "High",
                "description": "Contract manipulates balances after transfer"
            })
            
        # Check for overflow/underflow
        if "add" in str(contract_data) and not "SafeMath" in str(contract_data):
            vulnerabilities.append({
                "type": "Arithmetic Overflow",
                "severity": "Medium",
                "description": "Integer operations without SafeMath"
            })
            
        return {
            "vulnerabilities": vulnerabilities,
            "risk_level": "High" if any(v["severity"] == "High" for v in vulnerabilities) else 
                          "Medium" if any(v["severity"] == "Medium" for v in vulnerabilities) else "Low",
            "recommendation": "Audit recommended" if vulnerabilities else "Appears secure"
        }

class MultiversXAIAgent:
    def __init__(self):
        # Using a smaller model with lower token limits
        self.model = "llama3-8b-8192"
        self.max_tokens = 512
        self.last_request_time = 0
        
        # Initialize Satoshi modules
        self.consensus_analyzer = SatoshiConsensusAnalyzer()
        self.economic_modeler = SatoshiEconomicModeler()
        self.crypto_verifier = CryptographicVerifier()
        
    def extract_important_tx_data(self, tx_data):
        """Extract only the important fields from transaction data to reduce token usage"""
        if not tx_data:
            return {}
            
        important_fields = [
            "txHash", "receiver", "sender", "value", "fee", 
            "status", "timestamp", "data", "function"
        ]
        
        return {k: tx_data.get(k) for k in important_fields if k in tx_data}
        
    def analyze_transaction(self, tx_data):
        # Extract only important transaction data to reduce token usage
        important_tx_data = self.extract_important_tx_data(tx_data)
        
        # Run cryptographic verification
        verification_result = "Cannot verify with available data"
        if "signature" in tx_data:
            signature_verified = self.crypto_verifier.verify_transaction_signature(tx_data)
            verification_result = "Signature verification: " + ("Passed" if signature_verified else "Failed")
            
        # Calculate transaction entropy (randomness as measure of abnormality)
        tx_str = json.dumps(important_tx_data)
        entropy = 0
        if tx_str:
            # Shannon entropy calculation
            char_counts = defaultdict(int)
            for char in tx_str:
                char_counts[char] += 1
            
            str_len = len(tx_str)
            entropy = -sum((count/str_len) * np.log2(count/str_len) for count in char_counts.values())
        
        # Add Satoshi analysis results to the prompt
        prompt = f"""
        Analyze this MultiversX blockchain transaction and provide brief insights:
        {json.dumps(important_tx_data, indent=2)}
        
        Cryptographic verification: {verification_result}
        Transaction entropy: {entropy:.2f} (higher values may indicate complex or unusual transactions)
        
        Please provide in less than 200 words:
        1. Transaction type and purpose
        2. Risk assessment (including double-spend risk)
        3. Anomaly detection (is this transaction unusual?)
        4. Economic implications (if applicable)
        """
        
        # Rate limiting
        self._respect_rate_limits()
        
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            self.last_request_time = time.time()
            return completion.choices[0].message.content
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                st.warning("Rate limit exceeded. Please try again in a minute.")
                time.sleep(5)  # Add small delay
            return f"Error analyzing transaction: {str(e)}"
    
    def get_market_insights(self, market_data):
        # Extract only important market data
        important_market_data = self._extract_important_market_data(market_data)
        
        # Add economic analysis
        economic_analysis = self.economic_modeler.analyze_token_economics(market_data)
        
        # Get consensus health - ensure blocks data is a list
        blocks_data = market_data.get("blocks", [])
        # Convert to list if it's an integer (count of blocks) rather than actual blocks data
        if isinstance(blocks_data, int):
            blocks_data = []
            
        consensus_health = self.consensus_analyzer.detect_network_forks(blocks_data)
        
        satoshi_analysis = {
            "token_economics": economic_analysis,
            "consensus_health": consensus_health 
        }
        
        prompt = f"""
        Based on this MultiversX market data, provide brief strategic insights:
        {json.dumps(important_market_data, indent=2)}
        
        Satoshi's Analysis:
        - Token Velocity: {economic_analysis.get('token_velocity', 'N/A'):.2f} (lower is better for store of value)
        - Network Value Estimate: {economic_analysis.get('network_value', 'N/A'):,.0f}
        - Utility vs Speculation Ratio: {economic_analysis.get('utility_ratio', 'N/A'):.2f} (higher means more utility)
        - Consensus Health: Fork Risk {consensus_health.get('fork_risk', 'Unknown')}
        - Nakamoto Coefficient: {consensus_health.get('nakamoto_coefficient', 'Unknown')} (higher is more decentralized)
        
        Please provide in less than 200 words:
        1. Market trend summary
        2. Trust and security assessment
        3. Economic incentive analysis
        4. Is the token capturing value effectively?
        """
        
        # Rate limiting
        self._respect_rate_limits()
        
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            self.last_request_time = time.time()
            
            # Combine AI analysis with Satoshi's analysis
            ai_analysis = completion.choices[0].message.content
            
            # Format Satoshi's key insights
            satoshi_insight = f"""
            💡 **Satoshi's Key Insights:**
            - Token health: {economic_analysis.get('economic_analysis', {}).get('value_capture', 'Unknown')}
            - Network security: {consensus_health.get('centralization_risk', 'Unknown')}
            - Staking pressure: {economic_analysis.get('staking_pressure', 'Unknown')}
            """
            
            return ai_analysis + "\n\n" + satoshi_insight
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                st.warning("Rate limit exceeded. Please try again in a minute.")
                time.sleep(5)  # Add small delay
            return f"Error generating market insights: {str(e)}"
            
    def _extract_important_market_data(self, market_data):
        """Extract only important market data to reduce token usage"""
        if not market_data:
            return {}
            
        # Only keep essential fields
        important_data = {}
        
        # Sample critical fields - adjust based on actual data structure
        if "price" in market_data:
            # Only keep the last few price points instead of entire history
            if isinstance(market_data["price"], list) and len(market_data["price"]) > 5:
                important_data["price"] = market_data["price"][-5:]
            else:
                important_data["price"] = market_data["price"]
                
        # Add other critical fields you want to keep
        for field in ["accounts", "blocks", "transactions", "supply"]:
            if field in market_data:
                important_data[field] = market_data[field]
                
        return important_data
    
    def _respect_rate_limits(self):
        """Simple rate limiting mechanism"""
        # Check time since last request
        time_since_last = time.time() - self.last_request_time
        
        # If we've made a request recently, add a delay
        if time_since_last < TOKEN_RESET_INTERVAL:
            wait_time = min(5, TOKEN_RESET_INTERVAL - time_since_last)
            st.info(f"Respecting rate limits. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
    def analyze_double_spend_risk(self, tx_data, recent_txs):
        """Analyze risk of double spending based on transaction patterns"""
        if not tx_data or not recent_txs:
            return {"risk": "Unknown", "confidence": 0}
            
        # Look for transactions from same sender in recent history
        sender = tx_data.get("sender", "")
        sender_recent_txs = [tx for tx in recent_txs if tx.get("sender") == sender]
        
        # Calculate time proximity
        timestamp = tx_data.get("timestamp", 0)
        time_diffs = [abs(tx.get("timestamp", 0) - timestamp) for tx in sender_recent_txs]
        min_time_diff = min(time_diffs) if time_diffs else float('inf')
        
        # Check for similar amounts (potential double spend)
        value = float(tx_data.get("value", 0))
        similar_value_txs = [tx for tx in sender_recent_txs 
                            if abs(float(tx.get("value", 0)) - value) / (value or 1) < 0.05]
        
        # Risk assessment
        risk_level = "Low"
        confidence = 0.7
        
        if min_time_diff < 60 and similar_value_txs:  # Within a minute
            risk_level = "High"
            confidence = 0.9
        elif min_time_diff < 300 and similar_value_txs:  # Within 5 minutes
            risk_level = "Medium"
            confidence = 0.8
            
        return {
            "risk": risk_level,
            "confidence": confidence,
            "similar_transactions": len(similar_value_txs),
            "min_time_difference": min_time_diff
        }

def fetch_multiversx_data(endpoint):
    try:
        response = requests.get(f"{MULTIVERSX_API}/{endpoint}")
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Initialize AI Agent
ai_agent = MultiversXAIAgent()

# Sidebar
st.sidebar.markdown('<div class="header-container"><span class="header-logo">🔍</span><span class="header-title"><span class="blockinsight-brand">BlockInsight</span> <span class="blockinsight-x">X</span></span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="font-size: 0.9rem; color: #666; margin-top: -15px; margin-bottom: 15px;">AI-Powered MultiversX Analytics</div>', unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown('<div class="sidebar-info">⚠️ <b>Rate Limits:</b> This app respects Groq API rate limits (5000 tokens/min)</div>', unsafe_allow_html=True)

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Transaction Analysis", "Market Insights", "Network Stats", "Satoshi's Dashboard"]
)

# Display information based on selected analysis type
if analysis_type == "Transaction Analysis":
    st.sidebar.markdown("### About Transaction Analysis")
    st.sidebar.info("Enter a transaction hash to get AI-powered insights on transaction purpose, risk assessment, and potential impacts.")
elif analysis_type == "Market Insights":
    st.sidebar.markdown("### About Market Insights")
    st.sidebar.info("Get AI analysis of market trends, token economics, and investment opportunities based on current MultiversX data.")
elif analysis_type == "Network Stats":
    st.sidebar.markdown("### About Network Stats")
    st.sidebar.info("Monitor real-time statistics about the MultiversX blockchain, including consensus health and validator distribution.")
elif analysis_type == "Satoshi's Dashboard":
    st.sidebar.markdown("### About Satoshi's Dashboard")
    st.sidebar.info("Advanced blockchain metrics inspired by Satoshi Nakamoto's principles, focusing on decentralization, economic incentives, and trust minimization.")

# Main content header
st.markdown('<div class="header-container"><span class="header-logo">🔍</span><span class="header-title"><span class="blockinsight-brand">BlockInsight</span> <span class="blockinsight-x">X</span><span class="header-subtitle">- AI Agent</span></span></div>', unsafe_allow_html=True)

if analysis_type == "Transaction Analysis":
    st.markdown('<div class="tab-header">Transaction Analysis</div>', unsafe_allow_html=True)
    
    tx_hash = st.text_input("Enter Transaction Hash", placeholder="Transaction hash...")
    
    if tx_hash:
        with st.spinner("Fetching transaction data..."):
            tx_data = fetch_multiversx_data(f"transactions/{tx_hash}")
            # Also fetch recent transactions for double-spend analysis
            recent_txs = fetch_multiversx_data("transactions?size=25")
        
        if tx_data:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.spinner("Analyzing transaction with AI..."):
                    analysis = ai_agent.analyze_transaction(tx_data)
                
                st.markdown("### AI Analysis")
                st.write(analysis)
            
            with col2:
                st.markdown("### Verification")
                
                # Perform double-spend risk analysis
                double_spend_risk = ai_agent.analyze_double_spend_risk(tx_data, recent_txs)
                
                # Display verification status with colored risk level
                risk_level = double_spend_risk['risk']
                risk_class = f"risk-{risk_level.lower()}" if risk_level in ["High", "Medium", "Low"] else ""
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Double-Spend Risk</div>
                    <div class="metric-value"><span class="{risk_class}">{risk_level}</span> (Confidence: {double_spend_risk['confidence']:.1f})</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Signature verification (conceptual)
                signature_valid = "signature" in tx_data
                if signature_valid:
                    st.success("✓ Signature Valid")
                else:
                    st.error("⚠ Cannot Verify Signature")
                
                # Display key transaction metrics
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Value</div>
                    <div class="metric-value">{tx_data.get("value", "N/A")}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Fee</div>
                    <div class="metric-value">{tx_data.get("fee", "N/A")}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display raw transaction data in expander
            with st.expander("View Key Transaction Details"):
                st.json(ai_agent.extract_important_tx_data(tx_data))

elif analysis_type == "Market Insights":
    st.markdown('<div class="tab-header">Market Insights</div>', unsafe_allow_html=True)
    
    # Fetch market data
    with st.spinner("Fetching market data..."):
        market_data = fetch_multiversx_data("stats")
    
    if market_data:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.spinner("Generating AI insights..."):
                insights = ai_agent.get_market_insights(market_data)
            
            st.markdown("### Market Analysis")
            st.write(insights)
        
        with col2:
            # Economic analysis
            economic_data = ai_agent.economic_modeler.analyze_token_economics(market_data)
            
            st.markdown("### Token Economics")
            
            velocity = economic_data.get('token_velocity', 0)
            velocity_delta = "-" if velocity < 5 else "+"
            velocity_color = "green" if velocity < 5 else "red"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Token Velocity</div>
                <div class="metric-value">{velocity:.2f} <span style="color:{velocity_color}">({velocity_delta})</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            utility = economic_data.get('utility_ratio', 0)
            utility_delta = "+" if utility > 0.5 else "-"
            utility_color = "green" if utility > 0.5 else "red"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Utility Ratio</div>
                <div class="metric-value">{utility:.2f} <span style="color:{utility_color}">({utility_delta})</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            value_capture = economic_data.get('economic_analysis', {}).get('value_capture', 'Unknown')
            value_color = "green" if value_capture == "High" else "orange" if value_capture == "Medium" else "red"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Value Capture</div>
                <div class="metric-value"><span style="color:{value_color}">{value_capture}</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create visualizations for simplified data
        if "price" in market_data and isinstance(market_data["price"], list):
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("Price History & Network Value")
            
            # Only use last 30 data points for visualization
            recent_data = market_data["price"][-30:] if len(market_data["price"]) > 30 else market_data["price"]
            df = pd.DataFrame(recent_data)
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df.iloc[:, 0] if df.shape[1] > 0 else [],
                name="EGLD Price",
                line=dict(color='blue')
            ))
            
            # Add network value line on secondary axis
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[economic_data.get('network_value', 0)/1e6] * len(df),  # Divide by 1M for scale
                name="Network Value Est. (millions)",
                line=dict(color='green', dash='dash'),
                yaxis="y2"
            ))
            
            # Set layout with secondary y-axis and improved styling
            fig.update_layout(
                title="EGLD Price vs Network Value",
                yaxis=dict(title="EGLD Price", titlefont=dict(color="blue")),
                yaxis2=dict(title="Network Value (mil)", titlefont=dict(color="green"), 
                           overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(245, 248, 250, 1)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "Network Stats":
    st.markdown('<div class="tab-header">Network Statistics</div>', unsafe_allow_html=True)
    
    with st.spinner("Fetching network statistics..."):
        stats = fetch_multiversx_data("stats")
        
    if stats:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tx_count = stats.get("transactions", "N/A")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{tx_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            account_count = stats.get("accounts", "N/A")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Active Accounts</div>
                <div class="metric-value">{account_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            block_count = stats.get("blocks", "N/A")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Blocks</div>
                <div class="metric-value">{block_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add consensus health analysis
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Consensus Health")
        
        # Get recent blocks for consensus analysis
        with st.spinner("Analyzing consensus health..."):
            recent_blocks = fetch_multiversx_data("blocks?size=50")
        
        if recent_blocks:
            # Convert to list if it's an integer
            if isinstance(recent_blocks, int):
                recent_blocks = []
                
            consensus_health = ai_agent.consensus_analyzer.detect_network_forks(recent_blocks)
            
            health_cols = st.columns(3)
            with health_cols[0]:
                fork_risk = consensus_health.get("fork_risk", "Unknown")
                fork_color = "green" if fork_risk == "Low" else \
                            "orange" if fork_risk == "Medium" else "red"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Fork Risk</div>
                    <div class="metric-value"><span style="color:{fork_color}">{fork_risk}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with health_cols[1]:
                central_risk = consensus_health.get("centralization_risk", "Unknown")
                central_color = "green" if central_risk == "Low" else \
                               "orange" if central_risk == "Medium" else "red"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Centralization Risk</div>
                    <div class="metric-value"><span style="color:{central_color}">{central_risk}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with health_cols[2]:
                nakamoto = consensus_health.get("nakamoto_coefficient", "Unknown")
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Nakamoto Coefficient</div>
                    <div class="metric-value">{nakamoto}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display top validators    
            if consensus_health.get("top_validators"):
                st.markdown("#### Top Block Producers")
                validator_df = pd.DataFrame(consensus_health["top_validators"], 
                                           columns=["Validator", "Blocks Produced"])
                st.dataframe(validator_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add more detailed stats in expandable sections
        with st.expander("More Network Stats"):
            st.json({k: stats[k] for k in stats if k not in ["price"]})

elif analysis_type == "Satoshi's Dashboard":
    st.markdown('<div class="tab-header">Satoshi\'s Chain Analytics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='satoshi-box'>
    <strong>Satoshi's Vision:</strong> In a truly decentralized system, trust is replaced by cryptographic verification. 
    This dashboard applies the core principles of Bitcoin to MultiversX analysis, 
    focusing on consensus health, economic incentives, and trust minimization.
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch all necessary data
    with st.spinner("Gathering blockchain data..."):
        stats = fetch_multiversx_data("stats")
        recent_blocks = fetch_multiversx_data("blocks?size=50")
        recent_txs = fetch_multiversx_data("transactions?size=100")
    
    if stats and recent_blocks:
        # Convert blocks to list if it's an integer
        if isinstance(recent_blocks, int):
            recent_blocks = []
            
        # Economic analysis
        economic_data = ai_agent.economic_modeler.analyze_token_economics(stats)
        consensus_health = ai_agent.consensus_analyzer.detect_network_forks(recent_blocks)
        
        # Layout
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Consensus Mechanisms")
            
            # Nakamoto coefficient visualization
            nakamoto = consensus_health.get("nakamoto_coefficient", 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Nakamoto Coefficient</div>
                <div class="metric-value">{nakamoto}</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">Higher is better - Number of validators needed to disrupt consensus</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(nakamoto/20, 1.0))  # Scale to 0-1 range (20 is ideal)
            
            # Centralization risk
            central_risk = consensus_health.get("centralization_risk", "Unknown")
            central_color = "green" if central_risk == "Low" else "orange" if central_risk == "Medium" else "red"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Centralization Risk</div>
                <div class="metric-value"><span style="color:{central_color}">{central_risk}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Block timing analysis
            if "timing_anomalies" in consensus_health:
                anomaly_count = len(consensus_health["timing_anomalies"])
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Block Timing Anomalies</div>
                    <div class="metric-value">{anomaly_count}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(1.0 - min(anomaly_count/10, 1.0))  # Inverse scale
                
            # Top validators pie chart
            if "top_validators" in consensus_health and consensus_health["top_validators"]:
                st.markdown("#### Block Producer Distribution")
                labels = [v[0][:10] + "..." if len(v[0]) > 10 else v[0] for v in consensus_health["top_validators"]]
                values = [v[1] for v in consensus_health["top_validators"]]
                
                # Add "Others" category
                total_blocks = sum(values)
                if total_blocks < 50:  # We fetched 50 blocks
                    labels.append("Others")
                    values.append(50 - total_blocks)
                
                colors = px.colors.qualitative.Plotly
                fig = px.pie(
                    values=values, 
                    names=labels, 
                    title="Block Producer Distribution",
                    color_discrete_sequence=colors,
                    hole=0.4
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=80)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Economic Incentives")
            
            # Token velocity
            velocity = economic_data.get("token_velocity", 0)
            velocity_status = 'Low - Good for Store of Value' if velocity < 4 else 'Medium' if velocity < 8 else 'High - Primarily Medium of Exchange'
            velocity_color = "green" if velocity < 4 else "orange" if velocity < 8 else "red"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Token Velocity</div>
                <div class="metric-value">{velocity:.2f}</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: {velocity_color};">{velocity_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Utility vs Speculation
            utility = economic_data.get("utility_ratio", 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Utility vs Speculation Ratio</div>
                <div class="metric-value">{utility:.2f}</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">Higher means more real-world utility vs speculative value</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(utility)  # 0-1 scale
            
            # Staking economics
            staking_ratio = economic_data.get("staking_ratio", 0)
            equilibrium = economic_data.get("equilibrium_staking", 0)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Staking Ratio</div>
                <div class="metric-value">{staking_ratio:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(staking_ratio)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Equilibrium Point</div>
                <div class="metric-value">{equilibrium:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(equilibrium)
            
            staking_pressure = economic_data.get('staking_pressure', 'Unknown')
            pressure_color = "green" if staking_pressure == "Increasing" else "red"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Staking Pressure</div>
                <div class="metric-value"><span style="color:{pressure_color}">{staking_pressure}</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Transaction patterns analysis
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("Transaction Patterns")
        
        if recent_txs:
            # Calculate transaction value distribution
            values = [float(tx.get("value", 0)) for tx in recent_txs if tx.get("value")]
            
            if values:
                # Create histogram of transaction values
                fig = px.histogram(
                    x=values, 
                    nbins=20, 
                    title="Transaction Value Distribution",
                    labels={"x": "Transaction Value"},
                    color_discrete_sequence=["#1f77b4"]
                )
                fig.update_layout(
                    xaxis_title="Transaction Value",
                    yaxis_title="Frequency",
                    plot_bgcolor='rgba(245, 248, 250, 1)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gini coefficient (measure of inequality in transaction values)
                values.sort()
                n = len(values)
                gini = sum((2 * i - n - 1) * values[i] for i in range(n)) / (n * sum(values)) if sum(values) > 0 else 0
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Transaction Value Gini Coefficient</div>
                    <div class="metric-value">{gini:.4f}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">Measures concentration of transaction values (higher = more concentration)</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(gini)
                
                # Transaction entropy
                tx_str = json.dumps(recent_txs)
                char_counts = defaultdict(int)
                for char in tx_str:
                    char_counts[char] += 1
                    
                str_len = len(tx_str)
                if str_len > 0:
                    entropy = -sum((count/str_len) * np.log2(count/str_len) for count in char_counts.values())
                    normalized_entropy = entropy / np.log2(len(char_counts))
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Transaction Entropy</div>
                        <div class="metric-value">{entropy:.2f}</div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">Measures randomness in transactions (higher = more diverse patterns)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(normalized_entropy)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('Built for the MultiversX AI MegaWave Hackathon 🚀')
st.markdown('<span style="font-weight: 700;">BlockInsight <span style="font-weight: 900;">X</span></span> - Bringing Satoshi\'s Vision to MultiversX Analytics', unsafe_allow_html=True)
st.caption("Using Groq LLM API • Fully respects rate limits • Powered by AI")
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---") 