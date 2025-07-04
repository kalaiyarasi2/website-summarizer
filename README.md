# 🌐 Enhanced Website Summarizer

An AI-powered tool to summarize websites, extract structured data, and generate detailed analytics — powered by Groq’s LLaMA 3.3-70B and MCP agents.

## 🔥 Features

- 🧠 Analyze any website using LLM prompts
- 🎯 Preset prompts (summary, product info, contact, technical stack, etc.)
- 🔄 Batch processing via text/CSV file
- 📊 Visual analytics dashboard (Plotly)
- 📥 Download results (JSON / TXT)
- 💡 Built with `Streamlit`, `pydantic_ai`, `asyncio`

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit**
- **Groq + LLaMA 3.3-70B**
- **MCPServer (pydantic_ai)**
- **Asyncio**
- **Plotly**
- **Pandas**

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/website-summarizer.git
cd website-summarizer
pip install -r requirements.txt
streamlit run main.py
