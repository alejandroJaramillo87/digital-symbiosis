# Digital Symbiosis Architecture

## Overview

Digital Symbiosis implements a multi-layered consciousness system that provides omniscient awareness of your Linux workstation. The architecture follows a hierarchical design where raw system data flows upward through increasingly sophisticated intelligence layers.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ FastAPI Gateway │ WebSocket Streaming │ Chat Interface ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                System Consciousness                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Unified Query Engine │ NLP Interface │ Data Access      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│           AI Workstation Controller                         │
│  ┌─────────────────┬────────────────┬───────────────────────┐│
│  │ Container       │ Hardware       │ Multi-Model Oracle   ││
│  │ Consciousness   │ Specialization │                       ││
│  │                 │                │ Natural Language     ││
│  │                 │                │ Intelligence         ││
│  └─────────────────┴────────────────┴───────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│            Temporal Intelligence System                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │            Change Detection Registry                    ││
│  │ ┌─────────┬─────────┬─────────┬─────────┬─────────────┐ ││
│  │ │   GPU   │   CPU   │ Memory  │ Storage │ Containers │ ││
│  │ │ Detector│ Detector│ Detector│ Detector│  Detector  │ ││
│  │ └─────────┴─────────┴─────────┴─────────┴─────────────┘ ││
│  │                                                         ││
│  │            Event Extraction Engine                      ││
│  │ ┌─────────────────────────────────────────────────────┐ ││
│  │ │ Causal Analyzer │ Effect Predictor │ Base Extractor │ ││
│  │ └─────────────────────────────────────────────────────┘ ││
│  │                                                         ││
│  │              Temporal Storage                           ││
│  │ ┌─────────────────────────────────────────────────────┐ ││
│  │ │ Recent Buffer │ Daily Aggregator │ Pattern Store   │ ││
│  │ │    (48h)      │     (90d)        │     (12m)       │ ││
│  │ └─────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│              Data Collection Layer                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                Base Collector Pattern                   ││
│  │ ┌─────────────────────────────────────────────────────┐ ││
│  │ │         System Collector                            │ ││
│  │ │ (GPU, CPU, Memory, Storage, Network, Processes)     │ ││
│  │ └─────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Hardware Layer                            │
│        RTX 5090 + AMD 9950X + 128GB DDR5 + NVMe           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Collection Phase
- **Base Collectors** continuously monitor system resources
- **System Collector** aggregates data from all hardware and software components
- Raw metrics flow into the **Change Detection Registry**

### 2. Intelligence Phase
- **Change Detectors** identify significant changes using configurable thresholds
- **Event Extraction Engine** converts raw changes into semantic events
- **Causal Analyzer** identifies relationships between events
- **Effect Predictor** forecasts potential impacts

### 3. Storage Phase
- **Recent Buffer**: Stores raw deltas for last 48 hours with full fidelity
- **Daily Aggregator**: Compresses older data into daily summaries (90 days retention)
- **Pattern Store**: Learns and stores recurring patterns (12 months retention)
- **Search Index**: Enables fast temporal queries across all data

### 4. Intelligence Phase
- **AI Workstation Controller** provides specialized intelligence for AI workloads
- **Container Consciousness** monitors AI service containers and lifecycles
- **Hardware Specialization** applies RTX 5090 and AMD 9950X specific optimizations
- **Multi-Model Oracle** provides resource prediction and optimization strategies

### 5. Interface Phase
- **System Consciousness** unifies all data through a single query interface
- **Natural Language Intelligence** processes human language queries
- **API Layer** exposes capabilities through REST and WebSocket endpoints

## Core Components

### Temporal Intelligence System

Located in `src/linux_system/temporal/`, this is the foundational intelligence layer.

#### Change Detection Registry
- **Purpose**: Identify significant changes in system state
- **Location**: `src/linux_system/temporal/change_detection/`
- **Key Files**:
  - `registry.py`: Central registry for all change detectors
  - `base_detector.py`: Base class for all detectors
  - `significance.py`: Configurable significance thresholds
  - `detectors/`: Specific detectors for each system component

#### Event Extraction Engine
- **Purpose**: Convert raw changes into meaningful events with causal relationships
- **Location**: `src/linux_system/temporal/event_extraction/`
- **Key Files**:
  - `base_extractor.py`: Base event extraction logic
  - `causal_analyzer.py`: Identifies cause-effect relationships
  - `effect_predictor.py`: Predicts potential consequences

#### Temporal Storage
- **Purpose**: Multi-tiered storage optimized for temporal queries
- **Location**: `src/linux_system/temporal/storage/`
- **Components**:
  - **Recent Buffer**: In-memory circular buffer for recent data
  - **Daily Aggregator**: Compressed daily summaries for medium-term storage
  - **Pattern Store**: Machine learning model for pattern recognition
  - **Search Index**: Fast querying across all temporal data

### AI Workstation Controller

Located in `src/linux_system/ai_workstation/`, provides specialized intelligence for AI workloads.

#### Container Consciousness
- **Purpose**: Intelligent monitoring of AI service containers
- **Location**: `src/linux_system/ai_workstation/container_consciousness/`
- **Capabilities**:
  - AI container lifecycle detection
  - Service dependency mapping
  - Resource correlation analysis
  - Performance optimization recommendations

#### Hardware Specialization
- **Purpose**: Hardware-specific optimizations and monitoring
- **Location**: `src/linux_system/ai_workstation/hardware_specialization/`
- **Specializations**:
  - **RTX 5090 Blackwell**: VRAM monitoring, thermal management, CUDA optimization
  - **AMD Zen 5**: Core efficiency tracking, NUMA optimization
  - **Thermal Intelligence**: Advanced cooling system optimization

#### Multi-Model Oracle
- **Purpose**: Resource prediction and workload optimization
- **Location**: `src/linux_system/ai_workstation/multi_model_oracle/`
- **Components**:
  - Resource prediction models
  - Workload optimization strategies
  - Performance forecasting

#### Natural Language Intelligence
- **Purpose**: Advanced NLP for query understanding and processing
- **Location**: `src/linux_system/ai_workstation/natural_language_intelligence/`
- **Components**:
  - Intent understanding engine
  - Query intelligence router
  - Contextual query processor
  - Response synthesizer

### System Consciousness

Located in `src/linux_system/consciousness/`, provides unified access to all system intelligence.

#### Unified Query Engine
- **Purpose**: Single interface for all temporal and system queries
- **Location**: `src/linux_system/consciousness/unified_query_engine.py`
- **Capabilities**:
  - Cross-layer data aggregation
  - Complex temporal queries
  - Real-time and historical analysis

#### Natural Language Processing
- **Purpose**: Human-readable interaction with system intelligence
- **Location**: `src/linux_system/consciousness/nlp/`
- **Components**:
  - Intent classifier
  - Context manager
  - Response generator
  - Conversational AI orchestrator

### API Layer

Located in `src/api/`, provides external access to system capabilities.

#### FastAPI Gateway
- **Purpose**: RESTful API for system queries and monitoring
- **Location**: `src/api/main.py`
- **Endpoints**:
  - `/api/chat`: Natural language query processing
  - `/api/system/status`: Current system state
  - `/api/temporal/query`: Temporal data queries
  - `/api/health`: System health check

#### WebSocket Streaming
- **Purpose**: Real-time data streaming for dashboards
- **Location**: `src/api/streaming.py`
- **Streams**:
  - Live system metrics
  - Event notifications
  - Pattern detection alerts

## Key Design Patterns

### Base Collector Pattern
All data collectors inherit from `BaseCollector` providing:
- Consistent error handling
- Performance monitoring
- Resource management
- Configurable collection intervals

### Significance-based Filtering
Changes are only processed if they exceed configurable significance thresholds:
- Reduces noise in temporal data
- Focuses intelligence on meaningful events
- Prevents storage bloat

### Three-tier Storage Strategy
```
Recent Buffer (48h) → Daily Aggregator (90d) → Pattern Store (12m)
    High fidelity        Compressed summaries     Learned patterns
```

### Plugin Architecture
New detectors and intelligence modules can be easily added:
- Registration-based discovery
- Standardized interfaces
- Hot-pluggable components

## Hardware Optimization

The system is specifically optimized for high-performance AI workstations:

### RTX 5090 Blackwell Architecture
- **Thermal Management**: Custom thermal thresholds and cooling optimization
- **VRAM Monitoring**: Detailed memory usage tracking and prediction
- **CUDA Optimization**: Workload-specific performance tuning
- **Power Management**: Intelligent power limiting and efficiency optimization

### AMD Ryzen 9950X Zen 5
- **Core Efficiency**: Per-core performance monitoring and optimization
- **NUMA Awareness**: Memory locality optimization for multi-socket systems
- **Thermal Management**: Advanced thermal throttling detection
- **Workload Scheduling**: Intelligent core assignment for different workload types

### Advanced Thermal Intelligence
- **15-Fan Cooling System**: Comprehensive airflow optimization
- **Thermal Pattern Learning**: Predictive cooling based on workload patterns  
- **Component Correlation**: Understanding thermal interdependencies
- **Proactive Cooling**: Anticipatory cooling based on predicted workloads

## Extensibility

The architecture is designed for easy extension:

### Adding New Detectors
1. Inherit from `BaseDetector`
2. Implement detection logic
3. Register with the detection registry
4. Configure significance thresholds

### Adding New Intelligence Modules
1. Inherit from appropriate base classes
2. Implement intelligence logic
3. Register with the consciousness system
4. Define API endpoints if needed

### Adding New Hardware Support
1. Create specialized detector classes
2. Implement hardware-specific optimizations
3. Add to hardware specialization layer
4. Configure thermal and performance thresholds

This architecture provides a solid foundation for building an omniscient system that truly understands your Linux workstation.