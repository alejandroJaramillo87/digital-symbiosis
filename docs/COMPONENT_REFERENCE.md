# Digital Symbiosis Component Reference

## Temporal Intelligence System

### Change Detection Registry
**Location**: `src/linux_system/temporal/change_detection/`

#### BaseDetector
- **File**: `base_detector.py`
- **Purpose**: Abstract base class for all change detectors
- **Key Methods**:
  - `detect_changes(old_snapshot, new_snapshot)`: Main change detection logic
  - `calculate_significance(change)`: Determine change importance
  - `get_supported_categories()`: List of categories this detector handles

#### Registry  
- **File**: `registry.py`
- **Purpose**: Central registration system for change detectors
- **Key Methods**:
  - `register_detector(detector)`: Add a new detector
  - `detect_all_changes(old_snapshot, new_snapshot)`: Run all detectors
  - `get_detectors_for_category(category)`: Get detectors for specific category

#### Specific Detectors

**GPU Detector** (`detectors/gpu_detector.py`)
- Monitors NVIDIA GPU metrics (temperature, utilization, VRAM)
- Detects thermal throttling, memory pressure, utilization spikes
- RTX 5090 optimized thresholds

**Process Detector** (`detectors/process_detector.py`)  
- Tracks process lifecycle (creation, termination, state changes)
- Monitors CPU and memory usage per process
- Detects process crashes and resource exhaustion

**Memory Detector** (`detectors/memory_detector.py`)
- System memory and swap monitoring
- Memory pressure detection
- Out-of-memory event identification

**Storage Detector** (`detectors/storage_detector.py`)
- Disk usage and I/O monitoring
- Health status tracking (SMART data)
- Performance degradation detection

**Network Detector** (`detectors/network_detector.py`)
- Network interface monitoring
- Traffic pattern analysis
- Connection state tracking

**Security Detector** (`detectors/security_detector.py`) 
- Security event monitoring
- Failed authentication attempts
- Suspicious activity detection

**Python Environment Detector** (`detectors/python_env_detector.py`)
- Virtual environment monitoring
- Package installation/removal tracking
- Environment activation detection

### Event Extraction Engine
**Location**: `src/linux_system/temporal/event_extraction/`

#### BaseExtractor
- **File**: `base_extractor.py`
- **Purpose**: Extract semantic events from raw changes
- **Key Methods**:
  - `extract_events(changes)`: Convert changes to events
  - `determine_severity(event)`: Calculate event severity
  - `generate_description(event)`: Human-readable event description

#### CausalAnalyzer
- **File**: `causal_analyzer.py`
- **Purpose**: Identify cause-effect relationships between events
- **Key Methods**:
  - `analyze_causality(events, time_window)`: Find causal relationships
  - `calculate_correlation(event1, event2)`: Correlation strength
  - `build_causal_graph(events)`: Create causality network

#### EffectPredictor
- **File**: `effect_predictor.py`
- **Purpose**: Predict potential consequences of events
- **Key Methods**:
  - `predict_effects(event)`: Forecast potential consequences
  - `calculate_probability(effect)`: Effect likelihood
  - `get_mitigation_strategies(effect)`: Suggested preventive actions

### Temporal Storage System
**Location**: `src/linux_system/temporal/storage/`

#### RecentBuffer
- **File**: `recent_buffer.py`
- **Purpose**: Circular buffer for recent data (48h retention)
- **Key Methods**:
  - `add_delta(delta)`: Store new system delta
  - `get_range(start_time, end_time)`: Query time range
  - `get_recent(hours)`: Get data from last N hours

#### DailyAggregator
- **File**: `daily_aggregator.py`
- **Purpose**: Compress data into daily summaries (90d retention)  
- **Key Methods**:
  - `aggregate_day(date)`: Create daily summary from recent buffer
  - `get_summaries_range(start_date, end_date)`: Query date range
  - `get_summary_statistics(date)`: Detailed statistics for specific day

#### PatternStore
- **File**: `pattern_store.py`
- **Purpose**: Learn and store recurring patterns (12m retention)
- **Key Methods**:
  - `learn_pattern(events)`: Extract pattern from event sequence
  - `match_pattern(events)`: Find matching known patterns
  - `get_confident_patterns()`: Retrieve high-confidence patterns

#### SearchIndex
- **File**: `search_index.py`
- **Purpose**: Fast searching across all temporal data
- **Key Methods**:
  - `index_delta(delta)`: Add delta to search index
  - `search(query)`: Full-text search across events
  - `find_similar(event)`: Find similar historical events

#### QueryEngine
- **File**: `query_engine.py`  
- **Purpose**: Unified query interface across all storage tiers
- **Key Methods**:
  - `execute_query(query)`: Execute temporal query
  - `execute_query_unified(query)`: Execute with unified result format
  - `get_statistics()`: Get storage and query statistics

### Core Types and Configuration
**Location**: `src/linux_system/temporal/`

#### Types
- **File**: `types.py`
- **Purpose**: Core data structures for temporal intelligence
- **Key Classes**:
  - `SystemSnapshot`: Complete system state at point in time
  - `SystemChange`: Individual change between snapshots
  - `SystemEvent`: Semantic event extracted from changes
  - `SystemDelta`: Collection of changes and events for a time period

#### Configuration
- **File**: `config.py`
- **Purpose**: System configuration optimized for RTX 5090 + AMD 9950X
- **Key Settings**:
  - Collection intervals and retention periods
  - Hardware-specific thresholds (thermal, performance)
  - Significance calculation parameters

#### Performance
- **File**: `performance.py`
- **Purpose**: Performance monitoring for temporal intelligence system
- **Key Methods**:
  - `measure_collection_time()`: Monitor collection performance
  - `measure_query_performance()`: Track query response times
  - `get_system_overhead()`: Calculate system resource usage

## AI Workstation Controller

### Container Consciousness
**Location**: `src/linux_system/ai_workstation/container_consciousness/`

#### AIContainerDetector
- **File**: `ai_container_detector.py`
- **Purpose**: Specialized detection for AI service containers
- **Capabilities**:
  - Docker container lifecycle monitoring
  - AI service specific metrics (inference latency, model loading)
  - Resource correlation between containers

#### ServiceLifecycleExtractor
- **File**: `service_lifecycle_extractor.py`
- **Purpose**: Extract service lifecycle events from container changes
- **Key Features**:
  - Service startup/shutdown detection
  - Health check monitoring
  - Dependency mapping

#### ContainerCorrelator
- **File**: `container_correlator.py`
- **Purpose**: Correlate container performance with system resources
- **Functionality**:
  - Cross-container resource analysis
  - Performance impact assessment
  - Optimization recommendations

### Hardware Specialization
**Location**: `src/linux_system/ai_workstation/hardware_specialization/`

#### RTX5090BlackwallDetector
- **File**: `rtx5090_blackwall_detector.py`
- **Purpose**: RTX 5090 Blackwell architecture specific monitoring
- **Features**:
  - Blackwell-specific metrics
  - Advanced thermal management
  - VRAM utilization patterns
  - CUDA optimization monitoring

#### AMDZen5Detector
- **File**: `amd_zen5_detector.py`
- **Purpose**: AMD Ryzen 9950X Zen 5 architecture monitoring
- **Features**:
  - Per-core performance tracking
  - NUMA efficiency monitoring
  - Thermal throttling detection
  - Workload optimization

#### ThermalIntelligenceDetector
- **File**: `thermal_intelligence_detector.py`
- **Purpose**: Advanced thermal management for 15-fan cooling systems
- **Capabilities**:
  - Multi-zone thermal monitoring
  - Predictive cooling algorithms
  - Thermal pattern recognition
  - Cooling optimization strategies

### Multi-Model Oracle
**Location**: `src/linux_system/ai_workstation/multi_model_oracle/`

#### ResourceOracle
- **File**: `resource_oracle.py`
- **Purpose**: Resource usage prediction and optimization
- **Functions**:
  - Resource demand forecasting
  - Capacity planning recommendations
  - Performance bottleneck identification

#### WorkloadPredictor
- **File**: `workload_predictor.py`
- **Purpose**: AI workload prediction and scheduling optimization
- **Features**:
  - Model inference time prediction
  - Optimal scheduling recommendations
  - Resource allocation strategies

#### StrategyEngine
- **File**: `strategy_engine.py`
- **Purpose**: Optimization strategy selection and execution
- **Capabilities**:
  - Multi-objective optimization
  - Strategy effectiveness tracking
  - Adaptive optimization algorithms

### Natural Language Intelligence
**Location**: `src/linux_system/ai_workstation/natural_language_intelligence/`

#### IntentUnderstandingEngine
- **File**: `intent_understanding_engine.py`
- **Purpose**: Advanced NLP for AI workstation queries
- **Features**:
  - ML-based intent classification
  - Entity extraction with spaCy integration
  - Temporal context understanding
  - Technical vocabulary comprehension

#### QueryIntelligenceRouter
- **File**: `query_intelligence_router.py`
- **Purpose**: Route queries to appropriate intelligence systems
- **Functions**:
  - Query complexity analysis
  - Optimal routing decisions
  - Load balancing across intelligence modules

#### ContextualQueryProcessor
- **File**: `contextual_query_processor.py`
- **Purpose**: Process queries with full context awareness
- **Capabilities**:
  - Multi-turn conversation handling
  - Context-aware query interpretation
  - Follow-up question generation

#### ResponseSynthesizer
- **File**: `response_synthesizer.py`
- **Purpose**: Generate human-readable responses from system data
- **Features**:
  - Technical data humanization
  - Visualization suggestions
  - Actionable recommendations

## System Consciousness

### Unified Query Engine
**Location**: `src/linux_system/consciousness/`

#### DataAccessLayer
- **File**: `data_access_layer.py`
- **Purpose**: Unified access to all system data sources
- **Methods**:
  - `get_temporal_data()`: Access temporal intelligence data
  - `get_current_state()`: Real-time system state
  - `get_ai_workstation_data()`: AI-specific metrics

#### UnifiedQueryEngine
- **File**: `unified_query_engine.py`  
- **Purpose**: Central query processing hub
- **Capabilities**:
  - Cross-system data aggregation
  - Complex query optimization
  - Result correlation and enrichment

#### SystemConsciousness
- **File**: `system_consciousness.py`
- **Purpose**: Main consciousness coordination
- **Functions**:
  - System-wide state management
  - Intelligence orchestration
  - High-level decision making

### NLP Interface
**Location**: `src/linux_system/consciousness/nlp/`

#### IntentClassifier
- **File**: `intent_classifier.py`
- **Purpose**: Natural language intent classification
- **Features**:
  - Pattern-based intent recognition
  - Entity extraction
  - Temporal context parsing
  - Query parameter extraction

#### ConversationalAI
- **File**: `conversational_ai.py`
- **Purpose**: Complete conversational interface orchestration
- **Capabilities**:
  - Multi-step conversation handling
  - Context management across sessions
  - Response generation with system integration
  - Conversation analytics

## API Layer

### FastAPI Gateway
**Location**: `src/api/`

#### Main API
- **File**: `main.py`
- **Purpose**: FastAPI application entry point
- **Endpoints**:
  - `/api/chat`: Natural language query processing
  - `/api/system/status`: Current system state
  - `/api/temporal/query`: Temporal data queries
  - `/health`: System health check

#### Natural Language Processor
- **File**: `natural_language.py`
- **Purpose**: API layer for natural language processing
- **Function**:
  - Thin delegation layer to consciousness system
  - HTTP/API concerns handling
  - Response format conversion

#### Streaming Manager
- **File**: `streaming.py`
- **Purpose**: WebSocket streaming coordination
- **Features**:
  - Real-time metrics streaming
  - Event notification broadcasting
  - Connection management

## Data Collection Layer

### Base Collection Framework
**Location**: `src/linux_system/data_collection/`

#### BaseCollector
- **File**: `collectors/base_collector.py`
- **Purpose**: Abstract base for all data collectors
- **Features**:
  - Consistent error handling
  - Performance monitoring
  - Resource management
  - Configurable collection intervals

#### SystemCollector
- **File**: `collectors/system_collector.py`
- **Purpose**: Master collector orchestrating all system metrics
- **Responsibilities**:
  - Coordinating individual metric collectors
  - System snapshot generation
  - Collection scheduling
  - Data validation and sanitization

This component reference provides detailed information about each major component in the Digital Symbiosis system, including their purpose, location, key methods, and capabilities.