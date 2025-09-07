# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Digital Symbiosis is a comprehensive AI workstation consciousness system that provides intelligent monitoring, analysis, and optimization for high-performance AI hardware. The system is designed for RTX 5090 + AMD 9950X workstations and consists of two main parts:

1. **Core Intelligence System** (`src/`) - Python-based temporal intelligence and AI workstation control
2. **AI Experiments Infrastructure** (`ai-expirements/`) - Docker-based AI service orchestration (git submodule)

## Core Architecture

### Main Components

**Temporal Intelligence System** (`src/linux_system/temporal/`)
- **Change Detection**: Monitors system changes across GPU, processes, Python environments, memory, storage, security, and network
- **Event Extraction**: Converts raw changes into meaningful events with causal analysis
- **Storage**: Hierarchical storage system with recent buffer, daily aggregation, and pattern storage
- **Query Engine**: Natural language querying of temporal data

**AI Workstation Controller** (`src/linux_system/ai_workstation/`)
- **Container Consciousness**: AI container lifecycle management and resource correlation  
- **Hardware Specialization**: RTX 5090 Blackwell and AMD Zen 5 specific optimizations
- **Multi-Model Oracle**: Resource prediction and workload optimization
- **Natural Language Intelligence**: Intent understanding and contextual query processing

**System Consciousness** (`src/linux_system/consciousness/`)
- **Unified Query Engine**: Central query processing with NLP capabilities
- **Data Access Layer**: Abstracted access to all system data sources
- **Conversational AI**: Natural language interface to system intelligence

**API Layer** (`src/api/`)
- **FastAPI Gateway**: REST and WebSocket APIs for frontend integration
- **Real-time Streaming**: WebSocket-based system metrics streaming
- **Natural Language Processing**: Chat interface for system queries

### Key Design Patterns

- **Base Collector Pattern**: All data collectors inherit from `BaseCollector` with consistent error handling and performance monitoring
- **Temporal Storage**: Three-tier storage (recent buffer → daily aggregation → long-term patterns)
- **Change Detection Registry**: Pluggable detector system for different system components
- **Significance-based Filtering**: Only processes changes above configurable significance thresholds

## Development Commands

### AI Infrastructure (ai-expirements/)

The AI experiments infrastructure is managed via git submodule and uses Poetry + Docker:

```bash
# Navigate to AI infrastructure
cd ai-expirements/

# Install Python dependencies
make install          # poetry install
make update           # poetry update  
make shell            # poetry shell

# Docker services (GPU + CPU inference)
make up               # Start all services
make down             # Stop all services
make status           # Check service health
make gpu-up           # Start GPU services only
make cpu-up           # Start CPU services only
make ui-up            # Start Open WebUI only

# Development and testing
make test             # poetry run pytest -v
make lint             # poetry run ruff check .
make format           # poetry run black . && ruff format .

# Service monitoring
make logs             # All service logs
make logs-gpu         # GPU service logs only  
make logs-cpu         # CPU service logs only
make health           # Detailed health checks

# Quick start
make demo             # Start GPU + UI for demo
make dev              # Start CPU services for development
```

### Core System (src/)

The core Python system does not have a unified build system yet - test and run individual components directly:

```bash
# Run API server
cd src/
python -m api.main

# Run temporal collector (standalone)
python -m linux_system.temporal.collector

# Run AI workstation controller  
python -m linux_system.ai_workstation.ai_workstation_controller

# Test individual modules
python -m pytest tests/temporal/
```

## Service Endpoints

When the AI infrastructure is running:

- **CPU Inference**: http://localhost:8001-8003/v1 (load balanced)
- **GPU Inference**: http://localhost:8004/v1 (llama-gpu)  
- **vLLM GPU**: http://localhost:8005/v1 (high-performance)
- **Open WebUI**: http://localhost:3000 (chat interface)
- **API Gateway**: http://localhost:8000 (FastAPI backend)

## Configuration

### Temporal System Configuration

The temporal intelligence system is configured via `src/linux_system/temporal/config.py`:

- **RTX 5090 specific**: Thermal thresholds (80°C warning, 85°C critical, 90°C throttling)
- **AMD 9950X optimized**: 16-core tracking with NUMA efficiency monitoring
- **Collection interval**: 60 seconds (configurable)
- **Storage**: Recent buffer (48h), daily retention (90d), patterns (12m)

### Hardware Specialization

System is optimized for:
- **GPU**: NVIDIA RTX 5090 (32GB VRAM, Blackwell architecture)
- **CPU**: AMD Ryzen 9950X (16-core/32-thread, Zen 5)  
- **Memory**: 128GB DDR5-6000
- **Storage**: NVMe Gen 5 for models, separate OS drive

## Testing

```bash
# AI infrastructure tests (full test suite)
cd ai-expirements/
make test

# Core system tests (individual modules)  
cd src/
python -m pytest tests/temporal/test_process_detector.py
python -m pytest tests/temporal/test_python_env_detector.py
```

## Important Notes

- **Hardware Dependency**: System is optimized for RTX 5090 + AMD 9950X configuration
- **Thermal Management**: System includes advanced thermal monitoring and 15-fan cooling optimization
- **Security**: Container sandboxing with comprehensive security monitoring
- **Performance**: Designed for sustained AI workloads with proper resource management
- **Real-time**: WebSocket streaming for live system metrics and thermal data
- **Natural Language**: Chat interface supports questions like "Why did my GPU throttle?" or "Show me container patterns"