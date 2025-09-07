# Digital Symbiosis

An Omniscient System Intelligence for Your Localhost Linux Environment

Digital Symbiosis creates a comprehensive AI consciousness system that achieves complete awareness and understanding of your Linux workstation. Through sophisticated temporal intelligence, continuous monitoring, and natural language interaction, it becomes an omniscient companion that knows everything happening on your system.

## Vision

Transform your localhost Linux system into an intelligent, self-aware environment that:

- Knows Everything: Continuous monitoring of all system components, processes, containers, and hardware
- Understands Context: Temporal intelligence that learns patterns, correlations, and causality
- Speaks Naturally: Chat with your system using natural language queries
- Predicts and Optimizes: Proactive intelligence for performance optimization and issue prevention
- Learns Continuously: Self-improving system that gets smarter with every interaction

## What's Currently Implemented

This repository contains a sophisticated multi-layered intelligence system optimized for high-performance AI workstations (RTX 5090 + AMD 9950X):

### Core Intelligence System (src/)

**Temporal Intelligence System** (src/linux_system/temporal/)
- **Change Detection Registry**: Pluggable detectors for GPU, CPU, memory, storage, processes, containers, network, and security
- **Event Extraction Engine**: Converts raw system changes into meaningful events with causal analysis
- **Hierarchical Storage**: Recent buffer (48h) → daily aggregation (90d) → long-term patterns (12m)
- **Natural Language Query Engine**: Ask questions like "Why did my GPU throttle?" or "Show me yesterday's container patterns"

**AI Workstation Controller** (src/linux_system/ai_workstation/)
- Container Consciousness: Intelligent monitoring of AI service containers (llama-gpu, vllm, open-webui)
- Hardware Specialization: RTX 5090 Blackwell and AMD Zen 5 specific optimizations and monitoring
- Multi-Model Oracle: Resource prediction and workload optimization intelligence
- Natural Language Intelligence: Advanced NLP for intent understanding and contextual query processing

**System Consciousness** (src/linux_system/consciousness/)
- Unified Query Engine: Central intelligence hub with natural language processing
- Data Access Layer: Abstracted access to all temporal and system data
- Conversational AI: Full natural language interface with context management

**API Layer** (src/api/)
- FastAPI Gateway: REST and WebSocket APIs for frontend integration
- Real-time Streaming: WebSocket-based live system metrics and event streaming
- Chat Interface: Natural language processing endpoint for conversational queries

### AI Experiments Infrastructure (ai-expirements/)

Docker-based AI service orchestration managed via git submodule:
- GPU Inference Services: llama-gpu, vllm-gpu for high-performance inference
- CPU Inference Services: llama-cpu load balancer for CPU-based inference  
- Open WebUI: Chat interface for AI model interaction
- Orchestration: Poetry and Docker with comprehensive health monitoring

## Architecture Highlights

### Temporal Intelligence Design
- Base Collector Pattern: Consistent error handling and performance monitoring across all data collectors
- Significance-based Filtering: Only processes changes above configurable significance thresholds
- Three-tier Storage: Recent buffer → daily aggregation → long-term pattern recognition
- Causal Analysis: Automatic correlation and causality detection between system events

### Hardware Specialization
- RTX 5090 Optimization: Thermal management (80°C warning, 85°C critical, 90°C throttling), VRAM monitoring, Blackwell architecture features
- AMD 9950X Integration: 16-core tracking, NUMA efficiency monitoring, Zen 5 optimizations
- Advanced Thermal Intelligence: 15-fan cooling system optimization and thermal pattern learning

### Natural Language Intelligence
- Intent Classification: Advanced ML-based understanding of natural language queries
- Entity Extraction: Automatic identification of system components, time ranges, and query parameters
- Contextual Processing: Conversation memory and follow-up question understanding
- Response Synthesis: Human-readable explanations of complex system data and patterns

## Current Capabilities

- Real-time System Monitoring: GPU, CPU, memory, storage, network, thermal, processes, containers  
- Natural Language Queries: "Show me GPU temperature trends", "Why did my container restart?"  
- Temporal Pattern Recognition: Automatic detection of recurring system behaviors  
- Causal Analysis: Understanding why system events occur  
- AI Container Intelligence: Deep monitoring of inference services and model lifecycles  
- Predictive Capabilities: Resource usage prediction and optimization recommendations  
- WebSocket Streaming: Real-time dashboard and monitoring capabilities  
- Comprehensive Testing: Full test suite for temporal intelligence components  

## Getting Started

### Prerequisites
- Linux system (optimized for RTX 5090 + AMD 9950X but works on other configs)
- Docker and Docker Compose
- Python 3.9+ with Poetry
- Git with submodule support

### Quick Start

1. Clone with submodules:
   ```bash
   git clone --recursive https://github.com/yourusername/digital-symbiosis.git
   cd digital-symbiosis
   ```

2. Start AI Infrastructure (ai-expirements/):
   ```bash
   cd ai-expirements/
   make install     # Install Python dependencies
   make demo        # Start GPU + UI for demo
   # or make dev     # Start CPU services for development
   ```

3. Run Core Intelligence System:
   ```bash
   cd src/
   python -m api.main  # Start FastAPI server (http://localhost:8000)
   ```

4. Access Interfaces:
   - API Documentation: http://localhost:8000/docs
   - Open WebUI: http://localhost:3000
   - Chat Interface: Send natural language queries to `/api/chat`

### Example Queries

```
"What's my current GPU temperature?"
"Show me memory usage patterns from yesterday"  
"Why did my llama-gpu container restart?"
"Compare CPU vs GPU utilization over the last 4 hours"
"What thermal patterns do you see?"
"Predict when I might run out of VRAM"
"How are my containers performing?"
```

## Repository Structure

```
digital-symbiosis/
├── src/                          # Core Python intelligence system
│   ├── linux_system/
│   │   ├── temporal/            # Temporal intelligence and storage
│   │   ├── ai_workstation/      # AI workstation consciousness
│   │   ├── consciousness/       # Unified query engine & NLP
│   │   └── data_collection/     # System monitoring collectors
│   └── api/                     # FastAPI gateway and streaming
├── ai-expirements/              # AI service orchestration (submodule)
├── tests/                       # Comprehensive test suite
├── docs/                        # Documentation (to be expanded)
└── CLAUDE.md                    # Development guidance for Claude Code
```

## Hardware Optimization

This system is specifically optimized for high-performance AI workstations:

- RTX 5090 (32GB VRAM): Blackwall architecture monitoring, thermal intelligence, CUDA optimization
- AMD Ryzen 9950X: 16-core/32-thread monitoring, Zen 5 efficiency tracking, NUMA awareness
- 128GB DDR5-6000: Advanced memory pattern recognition and optimization
- NVMe Gen 5: High-speed model storage with intelligent caching patterns

## Development Philosophy

Following Linux philosophy principles:
- Do one thing well: Each component has a focused responsibility
- Composable: Components work together through well-defined interfaces
- Observable: Everything is logged, measured, and queryable
- Extensible: Plugin-based architecture for easy enhancement

## Future Enhancements

- Small LLM Integration: Replace rule-based NLP with fine-tuned small language models
- RAG Enhancement: Use temporal intelligence as knowledge base for advanced retrieval
- Learning Capabilities: Continuous improvement through user interaction patterns
- Advanced Prediction: Machine learning models for workload and resource forecasting

## Contributing

This is a learning and experimentation repository. Feel free to:
- Explore the temporal intelligence system
- Extend monitoring capabilities
- Improve natural language understanding
- Add new AI service integrations
- Enhance prediction and optimization algorithms

## License

MIT License - See LICENSE file for details.

---

*Digital Symbiosis: Where your Linux system becomes truly intelligent.*