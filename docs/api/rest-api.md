# REST API Reference

RTSM exposes a REST API for querying objects and system state.

**Base URL**: `http://localhost:8000`

---

## Objects

### List All Objects

```http
GET /objects
```

**Response**:

```json
[
  {
    "id": "a3f2c1",
    "label": "backpack",
    "xyz": [1.2, 0.4, 2.1],
    "confidence": 0.87,
    "hits": 15,
    "confirmed": true,
    "last_seen": "2024-01-15T10:30:00Z"
  }
]
```

**Query Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `confirmed_only` | bool | Only return confirmed objects (default: true) |
| `label` | string | Filter by label |
| `limit` | int | Max results (default: 100) |

---

### Get Object by ID

```http
GET /objects/{id}
```

**Response**:

```json
{
  "id": "a3f2c1",
  "label": "backpack",
  "xyz": [1.2, 0.4, 2.1],
  "confidence": 0.87,
  "hits": 15,
  "confirmed": true,
  "stability": 0.82,
  "view_count": 4,
  "last_seen": "2024-01-15T10:30:00Z"
}
```

---

### Get Object Image

```http
GET /objects/{id}/image
```

Returns the best JPEG crop of the object.

**Response**: `image/jpeg`

---

## Search

### Semantic Search

```http
GET /search/semantic?query={text}&top_k={n}
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural language query |
| `top_k` | int | Number of results (default: 5) |

**Example**:

```bash
curl "http://localhost:8000/search/semantic?query=red%20mug&top_k=5"
```

**Response**:

```json
{
  "query": "red mug",
  "results": [
    {
      "id": "b7d4e2",
      "label": "mug",
      "xyz": [0.8, 0.2, 1.5],
      "score": 0.82
    },
    {
      "id": "c9f1a3",
      "label": "cup",
      "xyz": [1.1, 0.3, 1.8],
      "score": 0.71
    }
  ]
}
```

---

### Spatial Search

```http
GET /search/spatial?x={x}&y={y}&z={z}&radius={r}
```

Find objects within a radius of a point.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x`, `y`, `z` | float | Center point (meters) |
| `radius` | float | Search radius (meters) |

**Response**:

```json
{
  "center": [1.0, 0.5, 2.0],
  "radius": 0.5,
  "results": [
    {
      "id": "a3f2c1",
      "label": "backpack",
      "xyz": [1.2, 0.4, 2.1],
      "distance": 0.24
    }
  ]
}
```

---

## System

### Health Check

```http
GET /health
```

**Response**:

```json
{
  "status": "ok",
  "uptime": 3600
}
```

---

### Statistics

```http
GET /stats/detailed
```

**Response**:

```json
{
  "frames_processed": 12450,
  "objects": {
    "total": 47,
    "confirmed": 32,
    "proto": 15
  },
  "memory": {
    "working_memory_mb": 128,
    "ltm_vectors": 32
  },
  "performance": {
    "avg_frame_ms": 28.5,
    "fps": 6.2
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "not_found",
  "message": "Object with ID 'xyz' not found"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (invalid parameters) |
| 404 | Object not found |
| 500 | Internal server error |

---

## Next Steps

- [WebSocket API](websocket.md) — Real-time streaming
- [Quick Start](../getting-started/quick-start.md) — Try these endpoints
