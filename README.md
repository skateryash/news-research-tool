# News Research Tool

🚀 **News Research Tool**: Revolutionizing Information Retrieval with AI 🔍

## Overview
The News Research Tool is an advanced application designed to streamline the process of extracting relevant information from vast news sources. Leveraging the latest in natural language processing (NLP) and vector search technologies, this tool offers researchers, journalists, and professionals a powerful means to enhance productivity and decision-making.

## Key Features
- **Advanced NLP Processing**: Utilizes state-of-the-art NLP models to preprocess and understand textual content.
- **Vector Embeddings**: Converts news articles into high-dimensional vectors using HuggingFace embeddings.
- **Efficient Vector Store**: Stores vectors in FAISS (Facebook AI Similarity Search) for rapid and accurate similarity searches.
- **Natural Language Queries**: Allows users to query the knowledge base with natural language, retrieving concise and relevant answers.
- **Source Attribution**: Provides links to the original news articles supporting the generated answers.

## Tech Stack
- **Backend**: LangChain framework
- **Frontend**: Streamlit interface
- **Libraries**: HuggingFace, FAISS, Google Gemini API

## Installation

### Prerequisites
- Python 3.9+
- pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/skateryash/news-research-tool.git
cd news-research-tool
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up API Keys
1. **Google Gemini API Key**: Obtain from [Google Gemini](https://makersuite.google.com/app/apikey).

Create a `.env` file in the project root and add your API keys:
```bash
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
```

## Usage
Run the application with Streamlit:
```bash
streamlit run app.py
```

### Using the Tool
1. **Input**: Provide URLs of relevant news sources.
2. **Processing**: The application will automatically load the content, split it into manageable chunks, and embed these chunks into vectors.
3. **Query**: Use natural language questions to query the stored knowledge base.
4. **Output**: Receive concise and relevant answers with links to the original news sources.


## Acknowledgements
- **Guided by**: Dhaval Patel
- **Libraries and Tools**: LangChain, Streamlit, HuggingFace, FAISS, Google Gemini

## Contact
For any questions or suggestions, please open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/yashgchaudhary/).
