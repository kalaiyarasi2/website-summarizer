import streamlit as st
import os
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import logging
from dataclasses import dataclass, asdict
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GROQ API key

import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file


@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    url: str
    prompt: str
    result: str
    processing_time: float
    timestamp: datetime
    word_count: int
    success: bool
    error_message: Optional[str] = None

class WebsiteSummarizer:
    """Enhanced Website Summarizer with advanced features"""
    
    def __init__(self):
        self.setup_mcp_server()
        self.setup_agent()
        self.analysis_history = []
        
    def setup_mcp_server(self):
        """Setup MCP Server with error handling"""
        try:
            self.mcp_fetch_server = MCPServerStdio(
                command='python',
                args=['-m', 'mcp_server_fetch']
            )
        except Exception as e:
            st.error(f"Failed to setup MCP server: {str(e)}")
            self.mcp_fetch_server = None
    
    def setup_agent(self):
        """Setup AI Agent with configuration"""
        if self.mcp_fetch_server:
            self.agent = Agent(
                model="groq:llama-3.3-70b-versatile",
                mcp_servers=[self.mcp_fetch_server]
            )
        else:
            self.agent = None
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def estimate_processing_time(self, url: str) -> str:
        """Estimate processing time based on URL"""
        domain = urlparse(url).netloc.lower()
        
        if any(x in domain for x in ['blog', 'article', 'news']):
            return "⚡ Fast (15-30 seconds)"
        elif any(x in domain for x in ['restaurant', 'menu', 'about']):
            return "🕐 Medium (30-60 seconds)"
        elif any(x in domain for x in ['shop', 'product', 'catalog']):
            return "⏳ Slow (60-120 seconds)"
        else:
            return "❓ Unknown (30-90 seconds)"
    
    async def analyze_website(self, url: str, prompt: str, timeout: int = 300) -> AnalysisResult:
        """Enhanced website analysis with error handling and timing"""
        start_time = time.time()
        
        try:
            if not self.agent:
                raise Exception("Agent not initialized")
                
            full_prompt = f"{prompt.strip()} {url.strip()}"
            
            async with self.agent.run_mcp_servers():
                result = await asyncio.wait_for(
                    self.agent.run(full_prompt), 
                    timeout=timeout
                )
                
            processing_time = time.time() - start_time
            word_count = len(result.output.split())
            
            analysis_result = AnalysisResult(
                url=url,
                prompt=prompt,
                result=result.output,
                processing_time=processing_time,
                timestamp=datetime.now(),
                word_count=word_count,
                success=True
            )
            
            self.analysis_history.append(analysis_result)
            return analysis_result
            
        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {timeout} seconds"
            logger.error(error_msg)
            return AnalysisResult(
                url=url,
                prompt=prompt,
                result="",
                processing_time=timeout,
                timestamp=datetime.now(),
                word_count=0,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return AnalysisResult(
                url=url,
                prompt=prompt,
                result="",
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                word_count=0,
                success=False,
                error_message=error_msg
            )

def create_sidebar():
    """Create enhanced sidebar with features"""
    st.sidebar.title("🛠️ Settings & Tools")
    
    # Analysis Settings
    st.sidebar.subheader("Analysis Settings")
    timeout = st.sidebar.slider("Timeout (seconds)", 60, 600, 300)
    
    # Preset Prompts
    st.sidebar.subheader("📝 Preset Prompts")
    preset_prompts = {
        "Website Summary": "Provide a comprehensive summary of this website including its purpose, main features, and key information",
        "Contact Info": "Extract all contact information including phone numbers, email addresses, and physical addresses",
        "Product Analysis": "Analyze the products or services offered, including features, pricing, and benefits",
        "Company Info": "Extract company information including mission, vision, team members, and business details",
        "Menu/Pricing": "Extract menu items, pricing, and service offerings in a structured format",
        "Technical Analysis": "Analyze the website's technical features, technologies used, and user experience elements"
    }
    
    selected_preset = st.sidebar.selectbox("Choose preset prompt:", list(preset_prompts.keys()))
    
    if st.sidebar.button("Use Preset Prompt"):
        st.session_state.selected_prompt = preset_prompts[selected_preset]
    
    return timeout, preset_prompts[selected_preset]

def create_analytics_dashboard(summarizer):
    """Create analytics dashboard"""
    if not summarizer.analysis_history:
        st.info("No analysis history available yet. Run some analyses to see statistics!")
        return
    
    st.subheader("📊 Analytics Dashboard")
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(result) for result in summarizer.analysis_history])
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(df))
    
    with col2:
        success_rate = (df['success'].sum() / len(df)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_time = df['processing_time'].mean()
        st.metric("Avg Processing Time", f"{avg_time:.1f}s")
    
    with col4:
        total_words = df['word_count'].sum()
        st.metric("Total Words Analyzed", f"{total_words:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing Time Chart
        fig_time = px.bar(
            df, 
            x='timestamp', 
            y='processing_time',
            title="Processing Time Over Time",
            color='success',
            color_discrete_map={True: 'green', False: 'red'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Word Count Distribution
        fig_words = px.histogram(
            df, 
            x='word_count',
            title="Word Count Distribution",
            nbins=20
        )
        st.plotly_chart(fig_words, use_container_width=True)
    
    # Success/Failure Analysis
    if df['success'].sum() != len(df):
        st.subheader("🔍 Error Analysis")
        failed_analyses = df[df['success'] == False]
        for _, row in failed_analyses.iterrows():
            st.error(f"**{row['url']}**: {row['error_message']}")

def create_batch_processing():
    """Create batch processing interface"""
    st.subheader("🔄 Batch Processing")
    
    # URL Input Methods
    input_method = st.radio("Choose input method:", ["Manual Entry", "File Upload"])
    
    urls = []
    if input_method == "Manual Entry":
        url_text = st.text_area("Enter URLs (one per line):", height=100, key="batch_urls")
        if url_text:
            urls = [url.strip() for url in url_text.split('\n') if url.strip()]
    
    else:  # File Upload
        uploaded_file = st.file_uploader("Upload CSV/TXT file with URLs", type=['csv', 'txt'], key="batch_file")
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                if uploaded_file.name.endswith('.csv'):
                    import io
                    df = pd.read_csv(io.StringIO(content))
                    if 'url' in df.columns:
                        urls = df['url'].tolist()
                    else:
                        st.error("CSV must have a 'url' column")
                else:
                    urls = [url.strip() for url in content.split('\n') if url.strip()]
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Batch Prompt
    batch_prompt = st.text_area("Enter prompt for all URLs:", height=100, key="batch_prompt")
    
    # Show URL count
    if urls:
        st.info(f"Found {len(urls)} URLs to process")
    
    return urls, batch_prompt

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Website Summarizer",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Enhanced Website Summarizer")
    st.markdown("*Powered by AI - Analyze websites with advanced features*")
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = WebsiteSummarizer()
    
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = ""
    
    summarizer = st.session_state.summarizer
    
    # Create sidebar
    timeout, preset_prompt = create_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Analysis", "🔄 Batch Processing", "📊 Analytics", "📚 History"])
    
    with tab1:
        st.subheader("Single Website Analysis")
        
        # URL input with validation
        url = st.text_input("🔗 Enter Website URL:", placeholder="https://example.com")
        
        if url and not summarizer.validate_url(url):
            st.error("❌ Please enter a valid URL with http:// or https://")
        
        # Prompt input
        prompt = st.text_area(
            "💬 Enter your prompt:",
            value=st.session_state.selected_prompt,
            height=100,
            placeholder="e.g., Summarize this website's main features and contact information"
        )
        
        # URL analysis preview
        if url and summarizer.validate_url(url):
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Domain:** {urlparse(url).netloc}")
            with col2:
                st.info(f"**Est. Time:** {summarizer.estimate_processing_time(url)}")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            custom_timeout = st.slider("Custom timeout (seconds):", 60, 600, timeout)
            include_metadata = st.checkbox("Include analysis metadata", value=True)
        
        # Submit button
        if st.button("🚀 Analyze Website", type="primary"):
            if url and prompt and summarizer.validate_url(url):
                with st.spinner("🔍 Analyzing website... This may take a few moments."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.1)
                        progress_bar.progress(i + 1)
                    
                    # Run analysis
                    result = asyncio.run(summarizer.analyze_website(url, prompt, custom_timeout))
                    
                    progress_bar.empty()
                    
                    if result.success:
                        st.success("✅ Analysis completed successfully!")
                        
                        # Display results
                        st.subheader("📋 Analysis Results")
                        st.write(result.result)
                        
                        # Metadata
                        if include_metadata:
                            with st.expander("📊 Analysis Metadata"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Processing Time", f"{result.processing_time:.2f}s")
                                with col2:
                                    st.metric("Word Count", result.word_count)
                                with col3:
                                    st.metric("Analysis Time", result.timestamp.strftime("%H:%M:%S"))
                        
                        # Export options
                        st.subheader("📤 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download as JSON
                            json_data = json.dumps(asdict(result), indent=2, default=str)
                            st.download_button(
                                "📁 Download JSON",
                                json_data,
                                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json"
                            )
                        
                        with col2:
                            # Download as Text
                            text_data = f"URL: {result.url}\nPrompt: {result.prompt}\nTimestamp: {result.timestamp}\n\nResults:\n{result.result}"
                            st.download_button(
                                "📄 Download Text",
                                text_data,
                                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain"
                            )
                    
                    else:
                        st.error(f"❌ Analysis failed: {result.error_message}")
                        
                        # Troubleshooting tips
                        st.subheader("🔧 Troubleshooting Tips")
                        st.write("- Check if the URL is accessible")
                        st.write("- Try a simpler prompt")
                        st.write("- Increase timeout duration")
                        st.write("- Check if the website allows scraping")
            
            else:
                st.error("❌ Please enter both a valid URL and prompt")
    
    with tab2:
        urls, batch_prompt = create_batch_processing()
        
        if st.button("🚀 Start Batch Processing", key="start_batch") and urls and batch_prompt:
            st.info(f"Processing {len(urls)} URLs...")
            
            # Process batch
            batch_results = []
            progress = st.progress(0)
            
            for i, url in enumerate(urls):
                with st.spinner(f"Processing {i+1}/{len(urls)}: {url}"):
                    result = asyncio.run(summarizer.analyze_website(url, batch_prompt))
                    batch_results.append(result)
                    progress.progress((i + 1) / len(urls))
            
            # Display batch results
            st.subheader("📊 Batch Results Summary")
            successful = sum(1 for r in batch_results if r.success)
            st.metric("Success Rate", f"{successful}/{len(batch_results)} ({successful/len(batch_results)*100:.1f}%)")
            
            # Results table
            results_df = pd.DataFrame([
                {
                    'URL': r.url,
                    'Status': '✅ Success' if r.success else '❌ Failed',
                    'Processing Time': f"{r.processing_time:.2f}s",
                    'Word Count': r.word_count,
                    'Error': r.error_message or ''
                }
                for r in batch_results
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Export batch results
            if batch_results:
                batch_json = json.dumps([asdict(r) for r in batch_results], indent=2, default=str)
                st.download_button(
                    "📁 Download Batch Results",
                    batch_json,
                    f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    key="download_batch"
                )
    
    with tab3:
        create_analytics_dashboard(summarizer)
    
    with tab4:
        st.subheader("📚 Analysis History")
        
        if summarizer.analysis_history:
            # Search and filter
            search_term = st.text_input("🔍 Search history:", placeholder="Search by URL or prompt...")
            
            filtered_history = summarizer.analysis_history
            if search_term:
                filtered_history = [
                    result for result in summarizer.analysis_history
                    if search_term.lower() in result.url.lower() or search_term.lower() in result.prompt.lower()
                ]
            
            # Display history
            for i, result in enumerate(reversed(filtered_history)):
                with st.expander(f"{'✅' if result.success else '❌'} {result.url[:50]}... - {result.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**URL:** {result.url}")
                    st.write(f"**Prompt:** {result.prompt}")
                    st.write(f"**Time:** {result.processing_time:.2f}s | **Words:** {result.word_count}")
                    
                    if result.success:
                        st.write("**Results:**")
                        st.write(result.result)
                    else:
                        st.error(f"**Error:** {result.error_message}")
            
            # Clear history button
            if st.button("🗑️ Clear History", key="clear_history"):
                summarizer.analysis_history = []
                st.rerun()
        
        else:
            st.info("No analysis history yet. Start analyzing websites to build your history!")

if __name__ == "__main__":
    main()