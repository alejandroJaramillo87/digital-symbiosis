# Digital Symbiosis Data Flow

This document explains how data flows through the Digital Symbiosis system, from raw hardware metrics to intelligent natural language responses.

## Overview

Data flows through five main phases:
1. **Collection** - Raw system metrics gathering
2. **Detection** - Significant change identification  
3. **Extraction** - Semantic event generation
4. **Intelligence** - Pattern recognition and analysis
5. **Interface** - Query processing and response generation

## Detailed Data Flow

### Phase 1: Collection
```
Hardware Layer
     ↓
Base Collectors (every 60s)
├── GPU metrics (temp, utilization, memory)
├── CPU metrics (usage, temp, frequency) 
├── Memory metrics (usage, swap, pressure)
├── Storage metrics (usage, I/O, health)
├── Process metrics (CPU, memory, lifecycle)
├── Container metrics (resources, status)
└── Network metrics (traffic, connections)
     ↓
System Collector
     ↓
Raw System Snapshot
```

**Key Files:**
- `src/linux_system/data_collection/collectors/system_collector.py`
- `src/linux_system/data_collection/collectors/base_collector.py`

### Phase 2: Detection
```
Raw System Snapshot
     ↓
Change Detection Registry
├── GPU Detector → GPU changes
├── Process Detector → Process changes
├── Memory Detector → Memory changes  
├── Storage Detector → Storage changes
├── Network Detector → Network changes
└── Security Detector → Security events
     ↓
Significance Filtering (configurable thresholds)
     ↓
Significant System Changes
```

**Key Files:**
- `src/linux_system/temporal/change_detection/registry.py`
- `src/linux_system/temporal/change_detection/detectors/`
- `src/linux_system/temporal/change_detection/significance.py`

### Phase 3: Extraction
```
Significant System Changes
     ↓
Event Extraction Engine
├── Base Extractor → Raw events
├── Causal Analyzer → Cause-effect relationships
└── Effect Predictor → Potential consequences
     ↓
Semantic Events with Causality
```

**Key Files:**
- `src/linux_system/temporal/event_extraction/base_extractor.py`
- `src/linux_system/temporal/event_extraction/causal_analyzer.py`
- `src/linux_system/temporal/event_extraction/effect_predictor.py`

### Phase 4: Storage & Intelligence
```
Semantic Events + Changes
     ↓
Temporal Storage (3-tier)
├── Recent Buffer (48h, full fidelity)
├── Daily Aggregator (90d, compressed)
└── Pattern Store (12m, learned patterns)
     ↓
AI Workstation Intelligence
├── Container Consciousness → Container insights
├── Hardware Specialization → Hardware optimization
├── Multi-Model Oracle → Predictions & recommendations
└── Natural Language Intelligence → Query understanding
     ↓
System Consciousness
```

**Key Files:**
- `src/linux_system/temporal/storage/`
- `src/linux_system/ai_workstation/`
- `src/linux_system/consciousness/`

### Phase 5: Interface
```
System Consciousness
     ↓
Unified Query Engine
├── Temporal queries → Historical data
├── Real-time queries → Current state
├── Pattern queries → Learned behaviors
└── Prediction queries → Future forecasts
     ↓
Natural Language Processing
├── Intent Classification → Query understanding
├── Context Management → Conversation memory
├── Response Generation → Human-readable answers
└── Conversational AI → Complete interaction
     ↓
API Layer
├── REST endpoints → Programmatic access
├── WebSocket streams → Real-time data
└── Chat interface → Natural language interaction
     ↓
User Interfaces
```

**Key Files:**
- `src/linux_system/consciousness/unified_query_engine.py`
- `src/linux_system/consciousness/nlp/`
- `src/api/`

## Data Types

### Raw System Snapshot
```python
SystemSnapshot = {
    "timestamp": datetime,
    "gpu": GPUMetrics,
    "cpu": CPUMetrics, 
    "memory": MemoryMetrics,
    "storage": StorageMetrics,
    "processes": List[ProcessInfo],
    "containers": List[ContainerInfo],
    "network": NetworkMetrics
}
```

### System Change
```python
SystemChange = {
    "timestamp": datetime,
    "category": str,  # "gpu", "cpu", "memory", etc.
    "entity_id": str,  # Specific component identifier
    "change_type": ChangeType,  # CREATED, UPDATED, DELETED
    "old_value": Any,
    "new_value": Any,
    "significance": float,  # 0.0 to 1.0
    "metadata": Dict[str, Any]
}
```

### System Event
```python
SystemEvent = {
    "timestamp": datetime,
    "event_type": str,  # "gpu_throttle", "process_crash", etc.
    "entity": str,  # Affected component
    "description": str,  # Human-readable description
    "confidence": float,  # 0.0 to 1.0
    "severity": EventSeverity,  # LOW, MEDIUM, HIGH, CRITICAL
    "related_changes": List[SystemChange],
    "predicted_effects": List[str],
    "causal_factors": List[str]
}
```

### System Delta
```python
SystemDelta = {
    "timestamp": datetime,
    "raw_delta": List[SystemChange],
    "semantic_events": List[SystemEvent], 
    "correlations": List[Correlation],
    "snapshot_metadata": Dict[str, Any]
}
```

## Storage Tiers

### Recent Buffer (48 hours)
- **Purpose**: High-fidelity recent data for detailed analysis
- **Storage**: In-memory circular buffer with disk persistence
- **Data**: Complete SystemDelta objects with all changes and events
- **Query Performance**: Instant access, millisecond response times

### Daily Aggregator (90 days)  
- **Purpose**: Compressed historical summaries for trend analysis
- **Storage**: Daily summary files with event compression
- **Data**: Aggregated statistics, significant events, pattern summaries
- **Query Performance**: Fast access, ~100ms response times

### Pattern Store (12 months)
- **Purpose**: Learned behavioral patterns and correlations
- **Storage**: Machine learning models and pattern definitions
- **Data**: Pattern templates, confidence scores, occurrence statistics  
- **Query Performance**: Pattern matching, ~500ms response times

## Query Types

### Real-time Queries
```python
# Current system status
query = TemporalQuery(
    result_type=QueryType.DELTAS,
    time_range=TimeRange.LAST_HOUR
)
```

### Historical Analysis
```python  
# Yesterday's GPU patterns
query = TemporalQuery(
    result_type=QueryType.EVENTS,
    categories=["gpu"],
    start_time=yesterday_start,
    end_time=yesterday_end
)
```

### Pattern Discovery
```python
# Recurring thermal events
query = TemporalQuery(
    result_type=QueryType.PATTERNS,
    pattern_types=["thermal"],
    min_pattern_confidence=0.8
)
```

### Natural Language Queries
```python
# "Why did my GPU throttle?"
query = ExtractedQuery(
    intent=QueryIntent.TROUBLESHOOTING,
    entities=[SystemEntity.GPU, SystemEntity.THERMAL],
    time_context=TimeContext(relative_time="recent"),
    requires_causal_analysis=True
)
```

## Performance Characteristics

### Collection Phase
- **Frequency**: 60 seconds (configurable)
- **Overhead**: <1% CPU impact
- **Latency**: Real-time data availability

### Detection Phase  
- **Processing Time**: <100ms per snapshot
- **Significance Filtering**: 70-90% noise reduction
- **Memory Usage**: ~10MB per hour of data

### Storage Phase
- **Recent Buffer**: ~1GB for 48 hours of data
- **Daily Aggregator**: ~50MB per day compressed
- **Pattern Store**: ~100MB for learned patterns

### Query Phase
- **Recent Data**: <10ms response time
- **Historical Data**: <100ms response time  
- **Pattern Matching**: <500ms response time
- **NLP Processing**: <2s for complex queries

## Scalability

### Horizontal Scaling
- **Multiple Collectors**: Can run specialized collectors on different systems
- **Distributed Storage**: Storage tiers can be distributed across nodes
- **Query Load Balancing**: Multiple query engines can share load

### Vertical Scaling
- **Memory**: More RAM enables larger recent buffers
- **Storage**: Faster SSDs improve historical query performance
- **CPU**: More cores enable parallel change detection

### Data Volume Handling
- **Compression**: 10:1 compression ratio for historical data
- **Pruning**: Automatic cleanup of low-significance data
- **Archiving**: Long-term pattern storage with minimal overhead

This data flow design ensures that the system can maintain complete awareness of your system state while providing fast, intelligent responses to both programmatic and natural language queries.