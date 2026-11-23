# Structured Data Extraction & Tool-Calling Telecom Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-green.svg)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.ai/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-OpenAI%20SDK-orange.svg)](https://ai.google.dev/)

This repository contains the complete implementation and evaluation notebook for **Module 20 Assignment 2**, covering two core paradigms of modern AI Engineering:
1. **Structured Data Extraction** using validated Pydantic schemas across Cloud (Gemini API via OpenAI SDK compatibility) and Local LLMs (Ollama `gemma4:e4b`).
2. **Tool-Calling Telecom Support Agent** using LangChain tool binding, dynamic routing, and an interactive CLI execution loop.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites & Installation](#prerequisites--installation)
- [Environment Configuration](#environment-configuration)
- [Project Structure & Modules](#project-structure--modules)
  - [Section 1: Structured Data Extraction](#section-1-structured-data-extraction)
  - [Section 2: Tool-Calling Telecom Support Agent](#section-2-tool-calling-telecom-support-agent)
  - [Section 3: Demonstration & Execution Logs](#section-3-demonstration--execution-logs)
- [Comparative Analysis (Cloud vs. Local)](#comparative-analysis-cloud-vs-local)
- [Verification & Validation Test Suite](#verification--validation-test-suite)
- [License](#license)

---

## 📌 Overview

As customer interactions scale, telecom operations require reliable automated processing for unstructured customer complaints as well as responsive, multi-turn AI agents capable of querying backend services.

This assignment addresses both challenges:
- **Module 1**: Converts noisy, unstructured user feedback (in English/Banglish) into strict, type-safe JSON payloads adhering to a business schema.
- **Module 2**: Constructs a Telecom Support Engineer AI agent equipped with custom tools to query user balances, list data packages, create support tickets, and check network statuses dynamically.

---

## ✨ Key Features

- **Pydantic V2 Schema Validation**: Strict field bounds (`urgency` constrained to $[1, 5]$), literal enum validation (`category`, `sentiment`), and default fallback handling.
- **Dual Inference Engine Support**:
  - **Cloud**: Google Gemini (`gemini-2.5-flash` / `gemini-1.5-flash`) via OpenAI-compatible endpoints.
  - **Local**: On-device runtime running Ollama (`gemma4:e4b`) inside Colab.
- **Function Calling & Agent Tools**: Custom `@tool` decorated functions with comprehensive type annotations and docstrings for automatic tool schema generation.
- **Dynamic Reasoning & Routing**: Single-turn and multi-turn tool orchestration with multi-tool calling capabilities.
- **Clean Execution Logs**: Post-processed clean log viewer preventing raw internal message dump noise.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────────────────────┐
                               │       Unstructured Input Text          │
                               └──────────────────┬─────────────────────┘
                                                  │
                        ┌─────────────────────────┴────────────────────────┐
                        │                                                  │
                        ▼                                                  ▼
      ┌───────────────────────────────────┐              ┌───────────────────────────────────┐
      │   Section 1: Data Extraction      │              │    Section 2: Support Agent       │
      ├───────────────────────────────────┤              ├───────────────────────────────────┤
      │ • Pydantic CustomerTicket Schema  │              │ • LangChain Tool Binding          │
      │ • Cloud: Gemini API (OpenAI SDK)  │              │ • Support Tools Engine            │
      │ • Local: Ollama (gemma4:e4b)      │              │ • Dynamic Tool Router & State     │
      └─────────────────┬─────────────────┘              └─────────────────┬─────────────────┘
                        │                                                  │
                        ▼                                                  ▼
      ┌───────────────────────────────────┐              ┌───────────────────────────────────┐
      │ Validated JSON Structured Output  │              │ Interactive Chat & Ticket Output │
      └───────────────────────────────────┘              └───────────────────────────────────┘
```

---

## ⚙️ Prerequisites & Installation

### Environment Setup (Google Colab Recommended)
The project is optimized for running within Google Colab (T4 GPU recommended for local Ollama execution).

```bash
# Install core dependencies
pip install -q pydantic openai langchain langchain-community langchain-core

# System tools for local Ollama execution
apt-get update && apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 🔑 Environment Configuration

The notebook securely pulls the Gemini API Key from Google Colab Secrets (`google.colab.userdata`) to ensure no hardcoded secrets exist in code.

Set up your secret key in Google Colab:
1. Open the **Secrets** tab (key icon) on the left sidebar.
2. Add a new secret with Name: `GEMINI_API_KEY`.
3. Paste your Gemini API key value and enable notebook access.

```python
from google.colab import userdata
import os

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

---

## 📦 Project Structure & Modules

### Section 1: Structured Data Extraction

#### 1.1 Pydantic Schema Definition (`CustomerTicket`)
```python
from pydantic import BaseModel, Field
from typing import Literal

class CustomerTicket(BaseModel):
    name: str = Field(description="Name of the customer, or 'Unknown' if missing")
    phone: str = Field(description="Phone number provided by customer, or 'Unknown'")
    location: str = Field(description="Customer location/area mentioned")
    category: Literal["Billing", "Network", "Package", "General"] = Field(...)
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(...)
    urgency: int = Field(ge=1, le=5, description="Urgency rating from 1 (lowest) to 5 (critical)")
    summary: str = Field(description="Concise summary of the ticket issue")
```

#### 1.2 Extraction Pipelines
- **Cloud Endpoint**: Uses `openai.OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")` passing `response_format=CustomerTicket`.
- **Local Endpoint**: Background server running `ollama serve` and pulling model `gemma4:e4b`, querying `http://localhost:11434/v1/`.

---

### Section 2: Tool-Calling Telecom Support Agent

#### 2.1 Defined Support Tools
1. `get_balance(phone_number: str)`: Fetches current main balance and active plan details.
2. `get_data_packages(category: str = 'all')`: Lists available internet/data packages.
3. `create_support_ticket(phone_number: str, issue_type: str, description: str)`: Submits an official ticket to backend IT.
4. `check_network_status(location: str)`: Inspects real-time tower maintenance/outages for a given area.

#### 2.2 Agent Architecture & Loop
- Utilizes `ChatOpenAI` bound with `bind_tools([...])`.
- Interactively routes user input through a `while True` CLI input loop.
- Automatically handles single tool queries, multi-tool queries (e.g., balance + network status in one turn), and plain conversational turns.

---

### Section 3: Demonstration & Execution Logs

Provides post-processed, clean execution summaries across:
- **Batch Data Extraction Comparison**: Cloud vs. Local validation matrix over 5 test cases.
- **Validation Prompts Execution**: Output logs for all 4 core test scenarios.
- **Multi-Turn Session Logs**: Transcript of an interactive session tracking user state.

---

## 📊 Comparative Analysis (Cloud vs. Local)

| Evaluation Metric | Cloud Model (Gemini API) | Local Model (Ollama `gemma4:e4b`) |
| :--- | :--- | :--- |
| **Average Latency** | ~0.8s – 1.4s | ~2.5s – 5.1s (hardware dependent) |
| **Schema Compliance** | 100% strict adherence | 95%+ adherence (occasional fallback handling needed) |
| **Name / Entity Fidelity** | High accuracy on Banglish names | High accuracy, slight variance on implied context |
| **Privacy & Cost** | Requires API quotas / cloud data transfer | Zero cost, 100% on-device data privacy |

---

## 🧪 Verification & Validation Test Suite

The agent and extraction pipelines were validated against the following standard queries:

1. **Balance Check Test**: `"Can you check the balance for my number 01716260866?"`
   - *Expected*: Calls `get_balance(phone_number='01716260866')`.
2. **Package Exploration Test**: `"What internet packages are available right now?"`
   - *Expected*: Calls `get_data_packages(category='all')`.
3. **Outage Verification Test**: `"I am facing internet issues in Dhanmondi. Is the network down?"`
   - *Expected*: Calls `check_network_status(location='Dhanmondi')`.
4. **Ticket Creation Test**: `"Please open a ticket for 01819223344 regarding billing error."`
   - *Expected*: Calls `create_support_ticket(...)` with appropriate metadata.

---

## 📜 License

This project is developed as part of the **AI Engineering Course (Module 20)**. Free to use for educational and research purposes.
