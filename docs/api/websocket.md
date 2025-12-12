# WebSocket API

RTSM provides a WebSocket endpoint for real-time streaming of point clouds and object updates.

**Endpoint**: `ws://localhost:8081`

---

## Connection

```javascript
const ws = new WebSocket('ws://localhost:8081');

ws.onopen = () => {
  console.log('Connected to RTSM');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};
```

---

## Message Types

### Point Cloud Update

Streamed periodically with the current point cloud.

```json
{
  "type": "point_cloud",
  "timestamp": "2024-01-15T10:30:00Z",
  "points": [
    [1.2, 0.4, 2.1, 255, 128, 64],
    [1.3, 0.5, 2.0, 240, 120, 60]
  ],
  "count": 10000
}
```

Each point is `[x, y, z, r, g, b]`.

---

### Objects Update

Sent when objects are created, updated, or removed.

```json
{
  "type": "objects_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "objects": [
    {
      "id": "a3f2c1",
      "label": "backpack",
      "xyz": [1.2, 0.4, 2.1],
      "confidence": 0.87,
      "confirmed": true
    }
  ]
}
```

---

### Object Created

Sent when a new object is first detected.

```json
{
  "type": "object_created",
  "object": {
    "id": "b7d4e2",
    "label": "mug",
    "xyz": [0.8, 0.2, 1.5],
    "confidence": 0.65,
    "confirmed": false
  }
}
```

---

### Object Confirmed

Sent when a proto-object is promoted to confirmed status.

```json
{
  "type": "object_confirmed",
  "object": {
    "id": "b7d4e2",
    "label": "mug",
    "xyz": [0.8, 0.2, 1.5],
    "confidence": 0.82,
    "confirmed": true
  }
}
```

---

### System Stats

Periodic system statistics.

```json
{
  "type": "stats",
  "timestamp": "2024-01-15T10:30:00Z",
  "fps": 6.2,
  "objects_count": 32,
  "frame_latency_ms": 28
}
```

---

## Client Commands

Send JSON commands to control the stream:

### Subscribe to Specific Events

```json
{
  "command": "subscribe",
  "events": ["objects_update", "stats"]
}
```

### Set Point Cloud Decimation

Reduce point cloud density for bandwidth:

```json
{
  "command": "set_decimation",
  "factor": 4
}
```

### Pause/Resume Streaming

```json
{
  "command": "pause"
}
```

```json
{
  "command": "resume"
}
```

---

## Example: Three.js Integration

```javascript
const ws = new WebSocket('ws://localhost:8081');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'point_cloud') {
    updatePointCloud(data.points);
  }

  if (data.type === 'objects_update') {
    updateObjectMarkers(data.objects);
  }
};

function updatePointCloud(points) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);

  points.forEach((p, i) => {
    positions[i * 3] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
    colors[i * 3] = p[3] / 255;
    colors[i * 3 + 1] = p[4] / 255;
    colors[i * 3 + 2] = p[5] / 255;
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Update your scene...
}
```

---

## Next Steps

- [REST API](rest-api.md) — Query endpoints
- [3D Demo source](https://github.com/calabi-inc/rtsm/tree/main/visualization) — Full Three.js example
