# RAGNAR

## Retrieval AuGmented kNowledge AdviseR

A powerful business intelligence agent that combines web research capabilities with database storage for comprehensive company and person analysis.

## 🚀 Features

- **Intelligent Business Research**: Automated research on companies and individuals using web search and AI analysis
- **Database-First Architecture**: Smart caching system that checks local database before performing web searches
- **Multi-LLM Support**: Compatible with multiple language model providers (Groq, OpenAI, Anthropic, etc.)
- **Interactive CLI**: User-friendly command-line interface with rich formatting
- **Comprehensive Tool Suite**: Six specialized tools for research and data management
- **Token Usage Tracking**: Built-in monitoring of LLM token consumption
- **LangSmith Integration**: Advanced tracing and debugging capabilities

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Tools](#tools)
- [Database Schema](#database-schema)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.13 or higher
- Supabase account for database storage
- API keys for LLM providers and web search

### Install from Source

```bash
git clone https://github.com/bgunyel/ragnar.git
cd ragnar
pip install -e .
```

### Dependencies

RAGNAR depends on several custom packages:
- `ai-common`: Common utilities for AI applications
- `business-researcher`: Web research and data extraction engine
- `supabase`: Database connectivity and operations

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# LLM Providers
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Web Search
TAVILY_API_KEY=your_tavily_api_key

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_supabase_secret_key

# Monitoring
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

## 🎯 Usage

### Interactive Mode

Start the interactive CLI:

```bash
python src/main_dev.py
```

Example conversation:
```
You: Research Apple Inc
Ragnar: [Provides comprehensive company information from database or web research]

You: Find information about Tim Cook at Apple
Ragnar: [Returns detailed person profile with professional background]

You: Insert the Apple research into database
Ragnar: [Stores company information in database for future queries]
```

### Programmatic Usage

```python
from ragnar import BusinessIntelligenceAgent
from ai_common import LlmServers

# Configure LLM settings
llm_config = {
    'language_model': {
        'model': 'llama-3.3-70b-versatile',
        'model_provider': LlmServers.GROQ.value,
        'api_key': 'your_api_key',
        'model_args': {
            'temperature': 0,
            'max_tokens': 32768
        }
    },
    'reasoning_model': {
        'model': 'llama-3.3-70b-versatile',
        'model_provider': LlmServers.GROQ.value,
        'api_key': 'your_api_key',
        'model_args': {
            'temperature': 0,
            'max_tokens': 32768
        }
    }
}

# Initialize agent
agent = BusinessIntelligenceAgent(
    llm_config=llm_config,
    web_search_api_key='your_tavily_key',
    database_url='your_supabase_url',
    database_key='your_supabase_key'
)

# Run research query
result = agent.run("Research Microsoft Corporation")
print(result['content'])
print(f"Tokens used: {result['token_usage']}")
```

## 🏗 Architecture

RAGNAR uses a modern agent architecture built on LangGraph with a clean inheritance hierarchy:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   LLM Call Node  │───▶│  Tools Call     │
└─────────────────┘    └──────────────────┘    │     Node        │
                                ▲               └─────────────────┘
                                │                        │
                                │                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Final Output  │◀───│ Conditional     │
                       └─────────────────┘    │ Decision        │
                                              └─────────────────┘
```

### Class Hierarchy

```
BaseAgent (Abstract)
├── Tool dispatcher pattern
├── LangGraph workflow management
├── State and memory handling
└── Token usage tracking

BusinessIntelligenceAgent (Concrete)
├── Inherits from BaseAgent
├── Business research capabilities
├── Database operations
└── 11 specialized tool handlers
```

### Key Components

- **BaseAgent**: Abstract base class with dispatcher pattern for tool handling
- **StateGraph**: Manages conversation flow and state transitions
- **Memory Saver**: Persists conversation history across sessions
- **Business Researcher**: Handles web search and content analysis
- **Supabase Client**: Manages database operations and caching
- **Tool Handlers**: Clean, testable methods for each tool operation
- **Token Tracker**: Monitors LLM usage and costs

## 🔧 Tools

RAGNAR provides six specialized tools:

### Research Tools
1. **ResearchPerson**: Research individuals with company context
2. **ResearchCompany**: Comprehensive company analysis

### Database Tools
1. **FetchCompanyFromDataBase**: Retrieve company records
2. **FetchPersonFromDataBase**: Retrieve person records
3. **InsertCompanyToDataBase**: Store company information
4. **InsertPersonToDataBase**: Store person information

### Smart Workflow

The agent implements a database-first approach:
1. Check local database for existing information
2. If found, return cached data instantly
3. If not found, perform web research
4. Optionally store results for future use

## 🗄 Database Schema

### Companies Table
```sql
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    industry VARCHAR,
    founded_year INTEGER,
    headquarters VARCHAR,
    website VARCHAR,
    employee_count INTEGER,
    revenue DECIMAL,
    created_by_id INTEGER,
    updated_by_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Persons Table
```sql
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    title VARCHAR,
    current_company_id INTEGER REFERENCES companies(id),
    bio TEXT,
    education TEXT,
    experience TEXT,
    skills TEXT[],
    created_by_id INTEGER,
    updated_by_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## 🔬 Development

### Project Structure

```
ragnar/
├── src/
│   ├── config.py                 # Configuration management
│   ├── main_dev.py              # Development CLI interface
│   └── ragnar/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base_agent.py               # Abstract base agent with dispatcher pattern
│       │   ├── business_intelligence_agent.py
│       │   ├── configuration.py
│       │   ├── enums.py
│       │   ├── state.py
│       │   ├── tools.py
│       │   └── utils.py
│       └── apps/
│           └── __init__.py
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
└── README.md
```

### Key Design Patterns

- **Agent Pattern**: Autonomous decision-making with tool selection
- **Dispatcher Pattern**: Clean tool handler mapping instead of large match-case blocks
- **Inheritance**: BaseAgent provides common functionality, child classes specialize
- **State Management**: Immutable state transitions via LangGraph
- **Dependency Injection**: Configurable LLM and database providers
- **Separation of Concerns**: Distinct modules for research, storage, and UI

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=ragnar tests/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation as needed

## 📊 Monitoring and Observability

RAGNAR includes comprehensive monitoring:

- **LangSmith Integration**: Request tracing and debugging
- **Token Usage Tracking**: Cost monitoring across models
- **Error Handling**: Graceful degradation and recovery
- **Performance Metrics**: Response time and throughput tracking

## 🔒 Security Considerations

- API keys stored in environment variables
- Database credentials encrypted at rest
- Input validation on all user inputs
- Rate limiting on external API calls
- Audit logging for database operations

## 📈 Performance Optimization

- **Database-first caching** reduces redundant web searches
- **Parallel processing** for multiple tool calls
- **Connection pooling** for database efficiency
- **Streaming responses** for improved user experience

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Verify all required environment variables are set
2. **Database Connection**: Check Supabase URL and key configuration
3. **Model Timeouts**: Adjust timeout settings in LLM configuration
4. **Memory Issues**: Monitor token usage and implement batching

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langgraph.com/)
- Database powered by [Supabase](https://supabase.com/)
- Web search via [Tavily](https://tavily.com/)
- CLI interface enhanced with [Rich](https://rich.readthedocs.io/)

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: bertan.gunyel@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/bgunyel/ragnar/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/bgunyel/ragnar/discussions)

---

**RAGNAR** - *Empowering business intelligence through AI-driven research and knowledge management.*