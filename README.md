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
