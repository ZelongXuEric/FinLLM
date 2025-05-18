FinLLM: Causal Reasoning in Finance with Large Language Models

This project explores the capabilities of LLMs in understanding and reasoning about causal relationships within the financial domain. Given the scarcity of dedicated causal datasets in finance, this project focuses on:

1.  Data Collection: Fetching financial news (Reuters), company filings (EDGAR), and macroeconomic data.
2.  Data Processing & Alignment: Cleaning textual data, identifying potential causal cues using keyword pre-screening, and aligning events with stock market reactions.
3.  Dataset Creation: Preparing data for manual annotation to build a new financial causality dataset.
4.  LLM Benchmarking: Evaluating the performance of LLMs (Qwen-3, Llama-3) on identifying causal and associative links in financial events based on human annotations.