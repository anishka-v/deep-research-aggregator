#!/usr/bin/env python3
"""
Deep Research Aggregator
A minimal tool to query multiple AI research sites and consolidate results.
College Project - Minimal Implementation
"""

import csv
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import requests


class Config:
    """Configuration loader from CSV file"""
    
    def __init__(self, csv_path: str = "sites_config.csv"):
        self.csv_path = csv_path
        self.sites = []
        
    def load(self) -> List[Dict]:
        """Load and parse configuration from CSV"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.sites = [row for row in reader if row.get('enabled', '').lower() == 'true']
            print(f"✓ Loaded {len(self.sites)} enabled sites from {self.csv_path}")
            return self.sites
        except FileNotFoundError:
            print(f"✗ Error: Configuration file '{self.csv_path}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            sys.exit(1)


# ==============================================================================
# Deep Research Provider Pattern (Extensible)
# ==============================================================================

class DeepResearchProvider(ABC):
    """Base class for deep research provider-specific logic"""
    
    @abstractmethod
    def get_headers(self, api_key: str) -> Dict[str, str]:
        """Get provider-specific HTTP headers"""
        pass
    
    @abstractmethod
    def build_payload(self, prompt: str) -> Dict:
        """Build provider-specific API payload"""
        pass
    
    @abstractmethod
    def get_timeout(self) -> int:
        """Get recommended timeout in seconds"""
        pass
    
    @abstractmethod
    def extract_urls_from_response(self, response: Dict) -> List[str]:
        """Extract URLs from provider-specific response format"""
        pass
    
    def get_research_prompt(self, user_prompt: str) -> str:
        """Default research prompt (can be overridden)"""
        return (
            f"Conduct comprehensive research on the following topic. "
            f"Provide detailed analysis with citations and URLs:\n\n"
            f"{user_prompt}\n\n"
            f"Include all source URLs."
        )


class PerplexityProvider(DeepResearchProvider):
    """Perplexity AI - Real-time search with citations"""
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    
    def build_payload(self, prompt: str) -> Dict:
        return {
            'model': 'sonar-pro',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a research assistant. Always cite sources with URLs.'
                },
                {
                    'role': 'user',
                    'content': self.get_research_prompt(prompt)
                }
            ],
            'temperature': 0.2,
            'search_domain_filter': [],
            'return_citations': True,
            'search_recency_filter': 'month'
        }
    
    def get_timeout(self) -> int:
        return 60  # 1 minute
    
    def extract_urls_from_response(self, response: Dict) -> List[str]:
        """Extract URLs from Perplexity response"""
        urls = []
        
        # Citations in dedicated field
        if 'citations' in response:
            urls.extend(response['citations'])
        
        # Extract from message content
        if 'choices' in response:
            for choice in response['choices']:
                content = choice.get('message', {}).get('content', '')
                urls.extend(re.findall(r'http[s]?://[^\s\)]+', content))
        
        return urls
    
    def get_research_prompt(self, user_prompt: str) -> str:
        return (
            f"Research the following topic thoroughly. "
            f"Search for recent, authoritative sources:\n\n"
            f"{user_prompt}\n\n"
            f"Provide:\n"
            f"- Comprehensive analysis\n"
            f"- Direct citations with URLs\n"
            f"- Recent sources (past year preferred)"
        )


class ChatGPTDeepResearchProvider(DeepResearchProvider):
    """OpenAI GPT-4 with enhanced deep research prompts"""
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'OpenAI-Beta': 'deep-research-v1'
        }
    
    def build_payload(self, prompt: str) -> Dict:
        return {
            'model': 'gpt-4',  # Use standard GPT-4 with research prompts
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are a research assistant conducting comprehensive deep research. '
                        'Your task is to search the web thoroughly, analyze multiple sources, '
                        'and provide well-researched answers with citations. '
                        'Always include direct URLs to all sources you reference. '
                        'Prioritize authoritative, recent, and relevant sources.'
                    )
                },
                {
                    'role': 'user',
                    'content': self.get_research_prompt(prompt)
                }
            ],
            'temperature': 0.2
            # Note: Only using standard OpenAI API parameters
            # Future parameters when dedicated API releases:
            # 'mode': 'deep_research', 'depth': 'advanced', etc.
        }
    
    def get_timeout(self) -> int:
        return 1800  # 30 minutes
    
    def extract_urls_from_response(self, response: Dict) -> List[str]:
        """Extract from Deep Research report format"""
        urls = []
        
        # Handle dedicated report format (if API uses this)
        if 'report' in response:
            report = response['report']
            
            # Sources section
            if 'sources' in report:
                for source in report['sources']:
                    if 'url' in source:
                        urls.append(source['url'])
            
            # Extract from markdown content
            if 'content' in report:
                urls.extend(re.findall(r'http[s]?://[^\s\)]+', report['content']))
        
        # Handle chat completion format (similar to OpenAI API)
        if 'choices' in response:
            for choice in response['choices']:
                content = choice.get('message', {}).get('content', '')
                urls.extend(re.findall(r'http[s]?://[^\s\)]+', content))
        
        # Handle citations field if provided
        if 'citations' in response:
            for citation in response['citations']:
                if isinstance(citation, dict) and 'url' in citation:
                    urls.append(citation['url'])
                elif isinstance(citation, str):
                    urls.append(citation)
        
        return urls
    
    def get_research_prompt(self, user_prompt: str) -> str:
        """Enhanced research prompt for deep research"""
        return (
            f"Conduct comprehensive research on the following topic:\n\n"
            f"{user_prompt}\n\n"
            f"Requirements:\n"
            f"1. Search multiple authoritative sources on the web\n"
            f"2. Analyze and synthesize information from diverse perspectives\n"
            f"3. Provide detailed findings with specific examples\n"
            f"4. Include direct URLs to all sources cited\n"
            f"5. Prioritize recent information (past 12 months)\n"
            f"6. Verify claims across multiple sources\n"
            f"7. Present a comprehensive report with clear citations\n\n"
            f"Format your response with:\n"
            f"- Executive summary\n"
            f"- Key findings with citations\n"
            f"- Detailed analysis\n"
            f"- List of all source URLs at the end"
        )


class GeminiDeepResearchProvider(DeepResearchProvider):
    """Google Gemini with Deep Thinking capability"""
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json'
            # Gemini uses API key in URL, not header
        }
    
    def build_payload(self, prompt: str) -> Dict:
        return {
            'contents': [{
                'parts': [{'text': self.get_research_prompt(prompt)}]
            }],
            'generationConfig': {
                'temperature': 0.2,
                'topP': 0.8,
                'topK': 40,
                'thinkingLevel': 'high'  # Enable deep reasoning
            },
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_ONLY_HIGH'
                }
            ]
        }
    
    def get_timeout(self) -> int:
        return 180  # 3 minutes
    
    def extract_urls_from_response(self, response: Dict) -> List[str]:
        """Extract from Gemini response format"""
        urls = []
        
        if 'candidates' in response:
            for candidate in response['candidates']:
                if 'content' in candidate:
                    parts = candidate['content'].get('parts', [])
                    for part in parts:
                        if 'text' in part:
                            urls.extend(re.findall(r'http[s]?://[^\s\)]+', part['text']))
        
        return urls
    
    def get_research_prompt(self, user_prompt: str) -> str:
        return (
            f"Using deep reasoning and analysis, research the following topic:\n\n"
            f"{user_prompt}\n\n"
            f"Instructions:\n"
            f"- Think step-by-step about the best approach to research this\n"
            f"- Consider multiple perspectives and sources\n"
            f"- Provide comprehensive analysis with specific URLs to sources\n"
            f"- Verify information from authoritative sources\n"
            f"- List all reference URLs clearly"
        )


class ProviderFactory:
    """Factory to create appropriate provider based on site name"""
    
    # Registry of providers
    PROVIDERS = {
        'perplexity': PerplexityProvider,
        'chatgpt deep research': ChatGPTDeepResearchProvider,
        'openai deep research': ChatGPTDeepResearchProvider,
        'gemini deep research': GeminiDeepResearchProvider,
        'gemini-3-deep-think': GeminiDeepResearchProvider,
    }
    
    @staticmethod
    def create(site_name: str) -> DeepResearchProvider:
        """Create provider based on site name"""
        site_name_lower = site_name.lower()
        
        # Check if site name matches any provider
        for key, provider_class in ProviderFactory.PROVIDERS.items():
            if key in site_name_lower:
                return provider_class()
        
        # Default to Perplexity format
        return PerplexityProvider()
    
    @staticmethod
    def register_provider(name: str, provider_class):
        """Allow registering new providers dynamically"""
        ProviderFactory.PROVIDERS[name.lower()] = provider_class


# ==============================================================================
# API Adapters
# ==============================================================================

class APIAdapter:
    """Base adapter for API calls"""
    
    def __init__(self, site_config: Dict):
        self.name = site_config['site_name']
        self.site_type = site_config['site_type']
        self.endpoint = site_config.get('api_endpoint', '')
        self.api_key = site_config.get('api_key', '')
        self.timeout = 30
        
    def query(self, prompt: str) -> Optional[Dict]:
        """Execute query and return response"""
        raise NotImplementedError


class ResearchAPIAdapter(APIAdapter):
    """Adapter for specialized research APIs (Tavily, Exa, etc.)"""
    
    def query(self, prompt: str) -> Optional[Dict]:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
            
            # Generic research API payload
            payload = {
                'query': prompt,
                'max_results': 20
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ✗ {self.name}: {str(e)}")
            return None


class LLMAPIAdapter(APIAdapter):
    """Adapter for LLM APIs used for research (OpenAI, Anthropic, etc.)"""
    
    def query(self, prompt: str) -> Optional[Dict]:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
            
            # Research-focused prompt
            research_prompt = (
                f"Research the following topic and provide relevant URLs and sources:\n\n"
                f"{prompt}\n\n"
                f"Please include direct URLs to articles, papers, and resources."
            )
            
            payload = {
                'model': 'gpt-4',  # Default, override in config
                'messages': [
                    {'role': 'user', 'content': research_prompt}
                ],
                'temperature': 0.7
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ✗ {self.name}: {str(e)}")
            return None


class DeepResearchLLMAdapter(APIAdapter):
    """Adapter for LLM-based deep research APIs - uses provider pattern"""
    
    def __init__(self, site_config: Dict):
        super().__init__(site_config)
        
        # Create provider-specific handler
        self.provider = ProviderFactory.create(site_config['site_name'])
        
        # Use provider-specific timeout
        self.timeout = self.provider.get_timeout()
    
    def query(self, prompt: str) -> Optional[Dict]:
        try:
            # Use provider-specific headers and payload
            headers = self.provider.get_headers(self.api_key)
            payload = self.provider.build_payload(prompt)
            
            # Handle API key in URL for Gemini
            endpoint = self.endpoint
            if 'gemini' in self.name.lower():
                endpoint = f"{self.endpoint}?key={self.api_key}"
                headers.pop('Authorization', None)
            
            print(f"  ⏳ {self.name}: Deep research in progress "
                  f"(timeout: {self.timeout}s)...")
            
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Use provider-specific URL extraction
            urls = self.provider.extract_urls_from_response(result)
            if urls:
                print(f"  📚 {self.name}: Found {len(urls)} URLs in response")
            
            return result
            
        except requests.Timeout:
            print(f"  ⏱️ {self.name}: Timeout after {self.timeout}s")
            return None
        except Exception as e:
            print(f"  ✗ {self.name}: {str(e)}")
            return None


class WebSearchAPIAdapter(APIAdapter):
    """Adapter for traditional web search APIs (Google, Bing, Brave, etc.)"""
    
    def query(self, prompt: str) -> Optional[Dict]:
        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.api_key if self.api_key else ''
            }
            
            params = {
                'q': prompt,
                'count': 20
            }
            
            response = requests.get(
                self.endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ✗ {self.name}: {str(e)}")
            return None


class ManualAdapter(APIAdapter):
    """Adapter for manually saved results"""
    
    def query(self, prompt: str) -> Optional[Dict]:
        """Load from pre-saved file"""
        try:
            manual_file = f"raw_results/{self.name.lower().replace(' ', '_')}_manual.json"
            if Path(manual_file).exists():
                with open(manual_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"  ⊘ {self.name}: Manual file not found, skipping")
                return None
        except Exception as e:
            print(f"  ✗ {self.name}: {str(e)}")
            return None


class AdapterFactory:
    """Factory to create appropriate adapter based on site type"""
    
    @staticmethod
    def create(site_config: Dict) -> APIAdapter:
        site_type = site_config.get('site_type', '').lower()
        
        if site_type == 'research_api':
            return ResearchAPIAdapter(site_config)
        elif site_type == 'llm_api':
            return LLMAPIAdapter(site_config)
        elif site_type == 'llm_deepresearch':
            return DeepResearchLLMAdapter(site_config)
        elif site_type == 'web_search_api':
            return WebSearchAPIAdapter(site_config)
        elif site_type == 'manual':
            return ManualAdapter(site_config)
        else:
            # Default to research API
            return ResearchAPIAdapter(site_config)


class URLExtractor:
    """Extract URLs from various response formats"""
    
    # Comprehensive URL regex pattern
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    @staticmethod
    def extract(data: any) -> Set[str]:
        """Recursively extract all URLs from any data structure"""
        urls = set()
        
        if isinstance(data, dict):
            for value in data.values():
                urls.update(URLExtractor.extract(value))
        elif isinstance(data, list):
            for item in data:
                urls.update(URLExtractor.extract(item))
        elif isinstance(data, str):
            # Find URLs in string
            found_urls = URLExtractor.URL_PATTERN.findall(data)
            urls.update(found_urls)
        
        return urls


class URLDeduplicator:
    """Deduplicate and normalize URLs"""
    
    # Parameters to remove from URLs
    TRACKING_PARAMS = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'msclkid', 'ref', 'source'
    }
    
    @staticmethod
    def normalize(url: str) -> str:
        """Normalize URL for comparison"""
        try:
            parsed = urlparse(url.lower().strip())
            
            # Remove www prefix
            netloc = parsed.netloc.replace('www.', '')
            
            # Remove tracking parameters
            query_params = parse_qs(parsed.query)
            filtered_params = {
                k: v for k, v in query_params.items() 
                if k not in URLDeduplicator.TRACKING_PARAMS
            }
            clean_query = urlencode(filtered_params, doseq=True)
            
            # Remove trailing slash from path
            path = parsed.path.rstrip('/')
            
            # Remove fragment
            normalized = urlunparse((
                'https',  # Standardize to https
                netloc,
                path,
                '',  # params
                clean_query,
                ''   # fragment
            ))
            
            return normalized
        except:
            return url
    
    @staticmethod
    def deduplicate(url_sources: Dict[str, List[str]]) -> List[Dict]:
        """Deduplicate URLs and rank by frequency"""
        # Map normalized URL to original URLs and sources
        url_map = {}
        
        for source, urls in url_sources.items():
            for url in urls:
                normalized = URLDeduplicator.normalize(url)
                
                if normalized not in url_map:
                    url_map[normalized] = {
                        'normalized': normalized,
                        'original_urls': set(),
                        'sources': set(),
                        'first_seen': source
                    }
                
                url_map[normalized]['original_urls'].add(url)
                url_map[normalized]['sources'].add(source)
        
        # Convert to list and sort by source count
        deduplicated = []
        for normalized, data in url_map.items():
            deduplicated.append({
                'url': normalized,
                'original_urls': list(data['original_urls']),
                'source_count': len(data['sources']),
                'sources': sorted(list(data['sources'])),
                'first_seen': data['first_seen']
            })
        
        # Sort by source count (descending)
        deduplicated.sort(key=lambda x: x['source_count'], reverse=True)
        
        return deduplicated


class ResearchAggregator:
    """Main orchestrator for the research aggregation"""
    
    def __init__(self, config_path: str = "sites_config.csv"):
        self.config = Config(config_path)
        self.sites = []
        self.raw_results_dir = Path("raw_results")
        self.raw_results_dir.mkdir(exist_ok=True)
        
    def run(self, query: str, max_workers: int = 10):
        """Execute research aggregation"""
        print(f"\n{'='*60}")
        print(f"Deep Research Aggregator")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Load configuration
        self.sites = self.config.load()
        
        if not self.sites:
            print("✗ No enabled sites found in configuration")
            return
        
        # Execute queries in parallel
        print(f"\n📡 Querying {len(self.sites)} sites in parallel...\n")
        start_time = time.time()
        
        results = self._execute_parallel_queries(query, max_workers)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed {len(results)} queries in {elapsed:.1f}s\n")
        
        # Save raw results
        print("💾 Saving raw results...")
        self._save_raw_results(results)
        
        # Extract URLs
        print("\n🔍 Extracting URLs from results...")
        url_sources = self._extract_urls(results)
        
        # Deduplicate
        print("\n🔗 Deduplicating and consolidating URLs...")
        deduplicated = URLDeduplicator.deduplicate(url_sources)
        
        # Save final output
        self._save_final_output(query, results, url_sources, deduplicated)
        
        # Print summary
        self._print_summary(url_sources, deduplicated)
        
    def _execute_parallel_queries(self, query: str, max_workers: int) -> Dict[str, any]:
        """Execute all queries in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_site = {}
            for site_config in self.sites:
                adapter = AdapterFactory.create(site_config)
                future = executor.submit(self._query_with_timing, adapter, query)
                future_to_site[future] = site_config['site_name']
            
            # Collect results as they complete
            for future in as_completed(future_to_site):
                site_name = future_to_site[future]
                try:
                    response, elapsed = future.result()
                    if response:
                        results[site_name] = response
                        print(f"  ✓ {site_name}: {elapsed:.1f}s")
                    else:
                        print(f"  ⊘ {site_name}: No response")
                except Exception as e:
                    print(f"  ✗ {site_name}: {str(e)}")
        
        return results
    
    def _query_with_timing(self, adapter: APIAdapter, query: str):
        """Query with timing"""
        start = time.time()
        response = adapter.query(query)
        elapsed = time.time() - start
        return response, elapsed
    
    def _save_raw_results(self, results: Dict[str, any]):
        """Save raw results to individual files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for site_name, data in results.items():
            filename = f"{site_name.lower().replace(' ', '_')}_{timestamp}.json"
            filepath = self.raw_results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(results)} raw result files to {self.raw_results_dir}/")
    
    def _extract_urls(self, results: Dict[str, any]) -> Dict[str, List[str]]:
        """Extract URLs from all results"""
        url_sources = {}
        
        for site_name, data in results.items():
            urls = URLExtractor.extract(data)
            if urls:
                url_sources[site_name] = list(urls)
                print(f"  {site_name}: {len(urls)} URLs")
        
        return url_sources
    
    def _save_final_output(self, query: str, results: Dict, url_sources: Dict, deduplicated: List[Dict]):
        """Save final consolidated output"""
        output = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_sources': len(results),
            'total_raw_links': sum(len(urls) for urls in url_sources.values()),
            'deduplicated_links': len(deduplicated),
            'links': deduplicated
        }
        
        output_file = 'final_links.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Final output saved to {output_file}")
    
    def _print_summary(self, url_sources: Dict, deduplicated: List[Dict]):
        """Print summary statistics"""
        total_raw = sum(len(urls) for urls in url_sources.values())
        
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Sources queried:     {len(url_sources)}")
        print(f"Total raw URLs:      {total_raw}")
        print(f"Deduplicated URLs:   {len(deduplicated)}")
        print(f"Reduction:           {(1 - len(deduplicated)/total_raw)*100:.1f}%")
        
        if deduplicated:
            print(f"\nTop 10 URLs by source count:")
            for i, link in enumerate(deduplicated[:10], 1):
                print(f"  {i}. [{link['source_count']}] {link['url'][:70]}")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python research_aggregator.py <query>")
        print("   or: python research_aggregator.py @sample_query.txt")
        sys.exit(1)
    
    # Get query from command line or file
    query_arg = sys.argv[1]
    if query_arg.startswith('@'):
        # Load from file
        query_file = query_arg[1:]
        try:
            with open(query_file, 'r') as f:
                query = f.read().strip()
        except FileNotFoundError:
            print(f"✗ Error: Query file '{query_file}' not found")
            sys.exit(1)
    else:
        query = query_arg
    
    # Run aggregator
    aggregator = ResearchAggregator()
    aggregator.run(query)


if __name__ == "__main__":
    main()

