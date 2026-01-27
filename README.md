# Deep Research Aggregator

A minimal Python tool to query multiple AI research sites/APIs in parallel, extract URLs from results, and produce a deduplicated consolidated list.

## Overview

This tool addresses the challenge of aggregating research results from multiple AI-powered research platforms. Instead of manually querying 50+ different sites, this script:

1. **Reads configuration** from a CSV file containing research sites and API credentials
2. **Executes queries in parallel** to maximize speed
3. **Extracts URLs** from all responses using pattern matching
4. **Deduplicates links** using algorithmic URL normalization
5. **Produces a consolidated list** ranked by frequency across sources

**Use Case:** When researching a topic, get a comprehensive list of relevant URLs by querying multiple AI research tools simultaneously.

## Configuration

### CSV Configuration File

The `sites_config.csv` file contains all site configurations. Format:

```csv
site_name,site_type,api_endpoint,api_key,enabled
Tavily,research_api,https://api.tavily.com/search,YOUR_API_KEY,false
Perplexity,llm_deepresearch,https://api.perplexity.ai/chat/completions,,false
```

### Examples

1. **Simple query**:
   ```bash
   python research_aggregator.py "What are the latest AI safety developments?"
   ```

### Site Types

- **`llm_deepresearch`**: LLM-based agentic research APIs (searches web)
  - Examples: Perplexity, ChatGPT Deep Research, Gemini Deep Think

- **`research_api`**: Specialized research APIs that return structured results with URLs
  - Examples: Tavily, Exa, Metaphor

- **`llm_api`**: Regular Large Language Model APIs (simple prompting)
  - Examples: OpenAI GPT-4, Claude, Gemini

- **`web_search_api`**: Traditional web search APIs
  - Examples: Google, Bing, Brave Search

 ## System Flow

```
┌─────────────────┐
│ sites_config.csv│
│ (Configuration) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Research Aggregator (Main)        │
│   - Config Parser                   │
│   - API Adapter Factory             │
│   - Parallel Query Orchestrator     │
└────────┬────────────────────────────┘
         │
         ├──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │ Site 1 │ │ Site 2 │ │ Site N │ │Manual  │
    │Research│ │  LLM   │ │ Web    │ │ Files  │
    │  API   │ │  API   │ │Search  │ │        │
    └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │          │
         ▼          ▼          ▼          ▼
    ┌────────────────────────────────────────┐
    │      raw_results/ (JSON files)         │
    └────────┬───────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  URL Extractor  │
    │  (Regex-based)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  URL Normalizer │
    │  & Deduplicator │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │final_links.json │
    └─────────────────┘
```


