# Deep Research Aggregator

A minimal Python tool to query multiple AI research sites/APIs in parallel, extract URLs from results, and produce a deduplicated consolidated list.


**Use Case:** When researching a topic, get a comprehensive list of relevant URLs by querying multiple AI research tools simultaneously.

## Algorithm Overview (Deep Research Aggregator)
- Load config (CSV): Read the list of ~50 research sources (site name, endpoint, API key, enabled flag).
- Adapter-based querying: Route each site through the correct API adapter (Research API / LLM / Web Search / Manual), all exposing a common query(prompt) interface.
- Parallel execution: Submit the same prompt to all enabled sites concurrently using ThreadPoolExecutor (configurable workers + timeouts + graceful failure).
- Save raw outputs: Store every site’s full response locally (e.g., raw_results/<site>.json) for transparency and debugging.
- Extract URLs: Recursively scan JSON/text responses and collect all HTTP/HTTPS links using regex + deep traversal.
- Normalize URLs: Canonicalize URLs (lowercase, strip www, remove tracking params/fragments, standardize HTTPS, remove trailing slashes).
- Deduplicate + consolidate: Group URLs by normalized form and track which sources mentioned each link.
- Rank results: Sort links by frequency (# of sources referencing each URL) to surface highest-consensus links.
- Generate final output: Produce a single combined, deduplicated list (e.g., final_links.json) including metadata (sources + counts).

## Configuration

### CSV Configuration File

The `sites_config.csv` file contains all 50 site configurations. Format:

```csv
site_name,site_type,api_endpoint,api_key,enabled
Tavily,research_api,https://api.tavily.com/search,YOUR_API_KEY,false
Perplexity,llm_deepresearch,https://api.perplexity.ai/chat/completions,,false
```
## Usage

### Basic Usage

```bash
python research_aggregator.py "Your research query here"
```

### Using Query from File

```bash
python research_aggregator.py @sample_query.txt

### Examples

1. **Simple query**:
   ```bash
   python research_aggregator.py "What are the latest AI safety developments?"
   ```
### Output

The script produces two types of output:

1. **Raw results**: Individual JSON files in `raw_results/` folder
   - Format: `{site_name}_{timestamp}.json`
   - Contains full API response for each site

2. **Final consolidated list**: `final_links.json`
   - Deduplicated URLs ranked by frequency
   - Includes metadata (source count, sources, etc.)

---

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


