/**
 * RTSM Unified Demo - Real-Time Spatio-Semantic Memory Visualization
 *
 * Render-only frontend that receives pre-computed meshes from Python backend.
 *
 * Features:
 * - Load static PLY point clouds from file
 * - Real-time point cloud streaming via WebSocket (pre-computed by backend)
 * - RTSM objects overlay with semantic labels
 * - Interactive 3D navigation with axis-constrained rotation
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

// ============================================================================
// SCENE SETUP
// ============================================================================

const app = document.getElementById('app')!

const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.shadowMap.enabled = false
app.appendChild(renderer.domElement)

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x111111)

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 2000)
camera.position.set(3, 3, 3)

const controls = new OrbitControls(camera, renderer.domElement)
controls.minPolarAngle = 0.0001
controls.maxPolarAngle = Math.PI - 0.0001
controls.target.set(0, 0, 0)
controls.update()

scene.add(new THREE.AmbientLight(0xffffff, 0.6))
const dir = new THREE.DirectionalLight(0xffffff, 0.6)
dir.position.set(3, 5, 2)
scene.add(dir)

// World container for axis flipping
const world = new THREE.Group()
scene.add(world)

// Axes helper
const axesHelper = new THREE.AxesHelper(0.5)
world.add(axesHelper)

// ============================================================================
// STATIC PLY LOADER
// ============================================================================

const loader = new PLYLoader()
let currentPoints: THREE.Points | null = null

function clearCurrentCloud() {
  if (currentPoints) {
    world.remove(currentPoints)
    currentPoints.geometry.dispose()
    ;(currentPoints.material as THREE.Material).dispose()
    currentPoints = null
  }
}

function fitCameraToGeometry(geometry: THREE.BufferGeometry) {
  geometry.computeBoundingBox()
  const bb = geometry.boundingBox
  if (!bb) return
  const size = new THREE.Vector3()
  bb.getSize(size)
  const radius = Math.max(size.x, size.y, size.z) * 0.5
  const dist = Math.max(1, radius * 2.5)
  camera.position.set(dist, dist, dist)
  camera.far = Math.max(2000, radius * 20)
  camera.updateProjectionMatrix()
  controls.target.set(0, 0, 0)
  controls.update()
  initialCamPos.copy(camera.position)
  initialTarget.copy(controls.target)
}

function loadPLYFile(file: File) {
  const url = URL.createObjectURL(file)
  loader.load(url, (geometry) => {
    URL.revokeObjectURL(url)
    clearCurrentCloud()
    const hasColor = !!geometry.getAttribute('color')
    geometry.computeBoundingBox()
    const bb = geometry.boundingBox
    if (bb) {
      const center = new THREE.Vector3()
      bb.getCenter(center)
      geometry.translate(-center.x, -center.y, -center.z)
    }
    let radius = 1
    if (bb) {
      const size = new THREE.Vector3()
      bb.getSize(size)
      radius = Math.max(size.x, size.y, size.z) * 0.5
    }
    const pointSize = Math.max(0.001, radius * 0.003)
    const material = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: hasColor,
      sizeAttenuation: true,
      depthWrite: false
    })
    currentPoints = new THREE.Points(geometry, material)
    currentPoints.frustumCulled = false
    world.add(currentPoints)
    fitCameraToGeometry(geometry)
  }, undefined, () => {
    URL.revokeObjectURL(url)
  })
}

// ============================================================================
// REAL-TIME POINT CLOUD STREAMING (WebSocket - Render Only)
// ============================================================================

// Binary message format (mesh_create):
// [magic:4 'PCLD'][mesh_id_len:2][mesh_id:N][num_points:4][positions:N*12][colors:N*3][has_pose:1][pose:64?]
const MAGIC_MESH_CREATE = 0x444C4350 // 'PCLD' little-endian
const MAGIC_CAMERA_FRAME = 0x464D4143 // 'CAMF' little-endian

const meshes = new Map<string, THREE.Points>()
let wsConnected = false
let meshCreateCount = 0

// ── Camera feed PiP (FaceTime-style: draggable, resizable, minimizable) ──
const cameraFeedImg = document.getElementById('camera-feed') as HTMLImageElement | null
const cameraPanel = document.getElementById('camera-panel') as HTMLElement | null
const cameraHeader = document.getElementById('camera-header') as HTMLElement | null
let cameraBlobUrl: string | null = null
let cameraFramePending = false
let cameraSizeLarge = false
let cameraRotated = true  // default: rotate 90° CW for portrait iPhone feeds
let cameraRotateScale = 0  // cached scale factor (0 = not yet computed)

// Camera minimize toggle
document.getElementById('camera-toggle')?.addEventListener('click', (e) => {
  e.stopPropagation()
  cameraPanel?.classList.toggle('collapsed')
})

// Camera resize toggle (small <-> large)
document.getElementById('camera-resize')?.addEventListener('click', (e) => {
  e.stopPropagation()
  if (!cameraPanel) return
  cameraSizeLarge = !cameraSizeLarge
  cameraPanel.style.width = cameraSizeLarge ? '400px' : '240px'
})

// Camera rotation toggle (landscape <-> portrait)
document.getElementById('camera-rotate')?.addEventListener('click', (e) => {
  e.stopPropagation()
  cameraRotated = !cameraRotated
  applyCameraRotation()
})

// Draggable camera panel
if (cameraPanel && cameraHeader) {
  let dragging = false
  let dragOffsetX = 0
  let dragOffsetY = 0

  cameraHeader.addEventListener('mousedown', (e) => {
    // Only start drag from the header bar itself, not buttons
    if ((e.target as HTMLElement).closest('.cam-btn')) return
    dragging = true
    dragOffsetX = e.clientX - cameraPanel.offsetLeft
    dragOffsetY = e.clientY - cameraPanel.offsetTop
    cameraPanel.style.transition = 'none'
    e.preventDefault()
  })

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return
    let x = e.clientX - dragOffsetX
    let y = e.clientY - dragOffsetY
    // Clamp to viewport
    x = Math.max(0, Math.min(x, window.innerWidth - cameraPanel.offsetWidth))
    y = Math.max(32, Math.min(y, window.innerHeight - cameraPanel.offsetHeight))
    cameraPanel.style.left = x + 'px'
    cameraPanel.style.top = y + 'px'
    cameraPanel.style.bottom = 'auto'
    cameraPanel.style.right = 'auto'
  })

  window.addEventListener('mouseup', () => {
    if (dragging) {
      dragging = false
      cameraPanel.style.transition = ''
    }
  })
}

// UI controls panel toggle (chevron)
const uiPanel = document.getElementById('ui')
document.getElementById('ui-toggle-row')?.addEventListener('click', () => {
  uiPanel?.classList.toggle('collapsed')
})

// Apply or remove camera rotation transform based on cached dimensions
function applyCameraRotation(): void {
  if (!cameraFeedImg) return
  if (cameraRotated && cameraRotateScale > 0) {
    cameraFeedImg.style.transform = `rotate(90deg) scale(${cameraRotateScale})`
    const marginPct = -((1 / cameraRotateScale) - cameraRotateScale) / 2 * 100
    cameraFeedImg.style.marginTop = marginPct + '%'
    cameraFeedImg.style.marginBottom = marginPct + '%'
  } else {
    cameraFeedImg.style.transform = ''
    cameraFeedImg.style.marginTop = ''
    cameraFeedImg.style.marginBottom = ''
  }
}

// Camera frame handler (binary CAMF protocol, Blob URL for native decode)
function handleCameraFrame(data: ArrayBuffer): void {
  if (!cameraFeedImg || cameraFramePending) return
  cameraFramePending = true

  const view = new DataView(data)
  const jpegLen = view.getUint32(4, true)
  const jpegData = new Uint8Array(data, 8, jpegLen)

  if (cameraBlobUrl) URL.revokeObjectURL(cameraBlobUrl)
  cameraBlobUrl = URL.createObjectURL(new Blob([jpegData], { type: 'image/jpeg' }))

  requestAnimationFrame(() => {
    if (!cameraFeedImg || !cameraBlobUrl) { cameraFramePending = false; return }
    cameraFeedImg.src = cameraBlobUrl

    // Compute rotation scale once from first loaded frame
    if (cameraRotateScale === 0) {
      cameraFeedImg.onload = () => {
        if (cameraRotateScale > 0 || !cameraFeedImg) return
        const w = cameraFeedImg.naturalWidth
        const h = cameraFeedImg.naturalHeight
        if (w > 0 && h > 0) {
          cameraRotateScale = w / h
          applyCameraRotation()
          cameraFeedImg.onload = null  // only need this once
        }
      }
    }

    cameraFramePending = false
  })
}
let poseUpdateCount = 0
let totalPoints = 0

function mat4FromRowMajorArray(a16: number[]): THREE.Matrix4 {
  // Three.js Matrix4.set() takes arguments in row-major order but stores column-major
  // Our backend sends row-major [r0c0, r0c1, r0c2, r0c3, r1c0, ...]
  // Three.js set() expects (r0c0, r0c1, r0c2, r0c3, r1c0, ...) - same order!
  // But internally stores column-major, so no transpose needed
  const m = new THREE.Matrix4()
  m.set(
    a16[0], a16[1], a16[2], a16[3],
    a16[4], a16[5], a16[6], a16[7],
    a16[8], a16[9], a16[10], a16[11],
    a16[12], a16[13], a16[14], a16[15],
  )
  // Don't transpose - set() already handles the conversion
  return m
}

function parseMeshCreate(buffer: ArrayBuffer): {
  meshId: string
  positions: Float32Array
  colors: Uint8Array
  pose: number[] | null
} | null {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)

  // Check magic
  const magic = view.getUint32(0, true)
  if (magic !== MAGIC_MESH_CREATE) {
    console.warn('[ws] Invalid mesh_create magic:', magic.toString(16))
    return null
  }

  // Parse header
  const meshIdLen = view.getUint16(4, true)
  const numPoints = view.getUint32(6, true)

  let offset = 10

  // mesh_id
  const meshIdBytes = bytes.slice(offset, offset + meshIdLen)
  const meshId = new TextDecoder().decode(meshIdBytes)
  offset += meshIdLen

  // positions (N * 3 * 4 bytes) - copy to ensure alignment
  const positionsSize = numPoints * 3 * 4
  const positionsBytes = bytes.slice(offset, offset + positionsSize)
  const positions = new Float32Array(positionsBytes.buffer, positionsBytes.byteOffset, numPoints * 3)
  offset += positionsSize

  // colors (N * 3 bytes)
  const colorsSize = numPoints * 3
  const colors = bytes.slice(offset, offset + colorsSize)
  offset += colorsSize

  // has_pose
  const hasPose = view.getUint8(offset)
  offset += 1

  // pose (optional)
  let pose: number[] | null = null
  if (hasPose) {
    pose = []
    for (let i = 0; i < 16; i++) {
      pose.push(view.getFloat32(offset + i * 4, true))
    }
  }

  return { meshId, positions, colors, pose }
}

function createOrUpdateMesh(
  meshId: string,
  positions: Float32Array,
  colors: Uint8Array,
  pose: number[] | null
) {
  let pts = meshes.get(meshId)

  if (!pts) {
    // Create new mesh
    const geom = new THREE.BufferGeometry()
    const mat = new THREE.PointsMaterial({
      size: 0.008,
      vertexColors: true,
      sizeAttenuation: true
    })
    pts = new THREE.Points(geom, mat)
    pts.frustumCulled = false
    pts.matrixAutoUpdate = false
    world.add(pts)
    meshes.set(meshId, pts)
    meshCreateCount++
  }

  // Update geometry
  pts.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

  // Convert colors from uint8 to float32 normalized
  const colorsFloat = new Float32Array(colors.length)
  for (let i = 0; i < colors.length; i++) {
    colorsFloat[i] = colors[i] / 255
  }
  pts.geometry.setAttribute('color', new THREE.BufferAttribute(colorsFloat, 3))
  pts.geometry.computeBoundingSphere()

  // Update point count
  totalPoints = 0
  for (const m of meshes.values()) {
    const posAttr = m.geometry.getAttribute('position')
    if (posAttr) totalPoints += posAttr.count
  }

  // Apply pose if provided
  if (pose) {
    const m = mat4FromRowMajorArray(pose)
    pts.matrix.copy(m)
    // Debug: log pose translation for first few meshes
    if (meshes.size <= 5) {
      console.log(`[debug-mesh] ${meshId} pose translation: [${pose[3].toFixed(3)}, ${pose[7].toFixed(3)}, ${pose[11].toFixed(3)}]`)
    }
  }
}

function updateMeshPose(meshId: string, pose: number[]) {
  const pts = meshes.get(meshId)
  if (!pts) {
    console.warn(`[ws] Pose update for unknown mesh: ${meshId}`)
    return
  }
  const m = mat4FromRowMajorArray(pose)
  pts.matrix.copy(m)
  poseUpdateCount++
}

function deleteMesh(meshId: string) {
  const pts = meshes.get(meshId)
  if (!pts) return

  world.remove(pts)
  pts.geometry.dispose()
  ;(pts.material as THREE.Material).dispose()
  meshes.delete(meshId)

  // Update point count
  totalPoints = 0
  for (const m of meshes.values()) {
    const posAttr = m.geometry.getAttribute('position')
    if (posAttr) totalPoints += posAttr.count
  }
}

function clearAllMeshes() {
  for (const [, pts] of meshes) {
    world.remove(pts)
    pts.geometry.dispose()
    ;(pts.material as THREE.Material).dispose()
  }
  meshes.clear()
  meshCreateCount = 0
  poseUpdateCount = 0
  totalPoints = 0
  updateHud()
}

let ws: WebSocket | null = null
let isManuallyDisconnected = false  // Flag to prevent auto-reconnect when user clicks Disconnect

function connectWebSocket() {
  // Connect to Python demo server via Vite proxy (or direct in production)
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${location.host}/ws`
  ws = new WebSocket(wsUrl)

  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    wsConnected = true
    console.log('[ws] Connected to demo server')
    updateHud()
    updateConnectionButtons()
  }

  ws.onclose = () => {
    wsConnected = false
    ws = null
    console.log('[ws] Disconnected')
    updateHud()
    updateConnectionButtons()
    // Don't auto-reconnect if user manually disconnected
    if (!isManuallyDisconnected) {
      console.log('[ws] Auto-reconnecting in 2s...')
      setTimeout(connectWebSocket, 2000)
    }
  }

  ws.onerror = () => {
    // Error handled by onclose
  }

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // Binary message - route by magic bytes
      if (ev.data.byteLength >= 4) {
        const magic = new DataView(ev.data).getUint32(0, true)
        if (magic === MAGIC_CAMERA_FRAME) {
          handleCameraFrame(ev.data)
        } else {
          // mesh_create (PCLD)
          const parsed = parseMeshCreate(ev.data)
          if (parsed) {
            createOrUpdateMesh(parsed.meshId, parsed.positions, parsed.colors, parsed.pose)
            updateHud()
          }
        }
      }
    } else {
      // JSON message
      try {
        const msg = JSON.parse(ev.data)

        if (msg.type === 'mesh_update_pose') {
          updateMeshPose(msg.mesh_id, msg.pose)
          updateHud()
        } else if (msg.type === 'mesh_delete') {
          deleteMesh(msg.mesh_id)
          updateHud()
        } else if (msg.type === 'clear') {
          clearAllMeshes()
        } else if (msg.type === 'stats') {
          console.log('[ws] Server stats:', msg)
        } else if (msg.type === 'objects_update') {
          // Real-time WM objects update from visualization server
          rtsmObjects = msg.objects || []

          // Debug: check if any objects have label_scores
          if (rtsmObjects.length > 0) {
            const withLabels = rtsmObjects.filter(o => o.label_scores && Object.keys(o.label_scores).length > 0)
            if (withLabels.length === 0) {
              console.warn('[ws] Objects received but none have label_scores - are you running the embedded RTSM visualization server?')
            }
          }

          updateObjectMarkers()
          updateObjectList()
          updateHud()
        } else if (msg.type === 'runtime_analytics') {
          handleAnalyticsMessage(msg)
        }
      } catch {
        // Ignore parse errors
      }
    }
  }
}

function sendCommand(cmd: string) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd }))
  }
}

// ============================================================================
// RTSM OBJECTS OVERLAY
// ============================================================================

interface RtsmObject {
  id: string
  xyz_world: [number, number, number] | number[] | null
  label_primary?: string | null
  label_hint?: string | null  // Secondary/unreliable label from CLIP - use id as primary
  label_scores?: Record<string, number>  // {label: score} for all candidate labels
  stability: number
  confirmed: boolean
}

const objectMarkers = new Map<string, THREE.Mesh>()
const objectLabels = new Map<string, THREE.Sprite>()
let rtsmObjects: RtsmObject[] = []
let showObjects = true
let showOnlyConfirmed = false

// Semantic search state (declared early so updateObjectMarkers can access)
let semanticSearchResults: Map<string, number> = new Map()  // object_id -> similarity score
let isSearchMode = false  // true when showing search results, false when showing all objects

// Selection highlight marker (always visible when object is selected)
let selectionMarker: THREE.Mesh | null = null
let selectionLabel: THREE.Sprite | null = null

// Threshold for picking display label from label_scores (lower than CLIP's 0.30)
const DISPLAY_LABEL_MIN_SCORE = 0.18

/**
 * Pick best display label for an object.
 * - If confirmed with non-unknown primary label, use it
 * - Otherwise pick highest scoring non-unknown label from label_scores above threshold
 * - Fallback to truncated ID
 */
function getBestDisplayLabel(obj: RtsmObject): string {
  // If we have a primary label that's not unknown, use it
  const primary = obj.label_primary || obj.label_hint
  if (primary && primary !== 'unknown') {
    return primary
  }

  // Check label_scores for best non-unknown label above threshold
  if (obj.label_scores) {
    let bestLabel = ''
    let bestScore = 0
    for (const [label, score] of Object.entries(obj.label_scores)) {
      if (label !== 'unknown' && score > bestScore && score >= DISPLAY_LABEL_MIN_SCORE) {
        bestLabel = label
        bestScore = score
      }
    }
    if (bestLabel) {
      return bestLabel
    }
  }

  // Fallback to truncated ID
  return obj.id.slice(0, 8)
}

/**
 * Get top-K labels with scores for display, sorted by score descending.
 */
function getTopKLabels(obj: RtsmObject, k = 3): Array<{ label: string; score: number }> {
  if (!obj.label_scores || Object.keys(obj.label_scores).length === 0) {
    return []
  }
  return Object.entries(obj.label_scores)
    .filter(([label]) => label !== 'unknown')
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
}

/**
 * Get all label names from an object (for search).
 */
function getAllLabelNames(obj: RtsmObject): string[] {
  const labels: string[] = []
  if (obj.label_primary) labels.push(obj.label_primary)
  if (obj.label && obj.label !== obj.label_primary) labels.push(obj.label)
  if (obj.label_scores) {
    labels.push(...Object.keys(obj.label_scores))
  }
  return labels
}

function createTextSprite(text: string, color = '#ffffff'): THREE.Sprite {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  canvas.width = 256
  canvas.height = 64
  ctx.fillStyle = 'rgba(0,0,0,0.6)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.font = 'bold 24px system-ui, Arial'
  ctx.fillStyle = color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, canvas.width / 2, canvas.height / 2)

  const texture = new THREE.CanvasTexture(canvas)
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true })
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(0.5, 0.125, 1)
  return sprite
}

function updateObjectMarkers() {
  // Remove old markers
  for (const [id, mesh] of objectMarkers) {
    if (!rtsmObjects.find(o => o.id === id)) {
      world.remove(mesh)
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
      objectMarkers.delete(id)
      const label = objectLabels.get(id)
      if (label) {
        world.remove(label)
        ;(label.material as THREE.SpriteMaterial).map?.dispose()
        ;(label.material as THREE.Material).dispose()
        objectLabels.delete(id)
      }
    }
  }

  // Debug: log first few object positions (only once when count changes)
  const objCount = rtsmObjects.filter(o => o.xyz_world).length
  if (objCount > 0 && objCount <= 5) {
    for (const obj of rtsmObjects.slice(0, 3)) {
      if (obj.xyz_world) {
        console.log(`[debug-obj] ${obj.id.slice(0, 8)} xyz: [${obj.xyz_world[0].toFixed(3)}, ${obj.xyz_world[1].toFixed(3)}, ${obj.xyz_world[2].toFixed(3)}]`)
      }
    }
  }

  // Add/update markers
  for (const obj of rtsmObjects) {
    if (!obj.xyz_world) continue

    let marker = objectMarkers.get(obj.id)
    if (!marker) {
      const geom = new THREE.SphereGeometry(0.03, 16, 16)
      const mat = new THREE.MeshBasicMaterial({
        color: obj.confirmed ? 0x00ff88 : 0xffaa00,
        transparent: true,
        opacity: 0.8
      })
      marker = new THREE.Mesh(geom, mat)
      marker.renderOrder = 999
      world.add(marker)
      objectMarkers.set(obj.id, marker)
    }

    marker.position.set(obj.xyz_world[0], obj.xyz_world[1], obj.xyz_world[2])
    // In search mode, only show objects that match the semantic search
    const matchesSearch = !isSearchMode || semanticSearchResults.has(obj.id)
    const visible = showObjects && (!showOnlyConfirmed || obj.confirmed) && matchesSearch
    marker.visible = visible
    ;(marker.material as THREE.MeshBasicMaterial).color.setHex(obj.confirmed ? 0x00ff88 : 0xffaa00)

    // Highlight selected object (larger scale)
    if (obj.id === selectedObjectId) {
      marker.scale.set(1.8, 1.8, 1.8)
    } else {
      marker.scale.set(1, 1, 1)
    }

    // Label: short ID + stability (no CLIP label — unreliable with current vocab)
    let label = objectLabels.get(obj.id)
    const labelText = `${obj.id.slice(0, 6)} ${obj.stability.toFixed(2)}`
    if (!label) {
      label = createTextSprite(labelText, obj.confirmed ? '#00ff88' : '#ffaa00')
      world.add(label)
      objectLabels.set(obj.id, label)
    } else {
      // Update label text if stability changed (recreate sprite)
      // Only recreate if text differs to avoid GC churn
      const currentText = (label.userData as any)?.text
      if (currentText !== labelText) {
        world.remove(label)
        ;(label.material as THREE.SpriteMaterial).map?.dispose()
        ;(label.material as THREE.Material).dispose()
        label = createTextSprite(labelText, obj.confirmed ? '#00ff88' : '#ffaa00')
        ;(label.userData as any) = { text: labelText }
        world.add(label)
        objectLabels.set(obj.id, label)
      }
    }
    label.position.set(obj.xyz_world[0], obj.xyz_world[1] + 0.08, obj.xyz_world[2])
    label.visible = visible
  }

  // Update selection highlight marker (always visible when selected)
  updateSelectionMarker()
}

/**
 * Update the selection highlight marker.
 * This marker is always visible when an object is selected, even if objects are hidden.
 */
function updateSelectionMarker() {
  const selectedObj = selectedObjectId ? rtsmObjects.find(o => o.id === selectedObjectId) : null

  if (!selectedObj || !selectedObj.xyz_world) {
    // No selection - hide marker
    if (selectionMarker) {
      selectionMarker.visible = false
    }
    if (selectionLabel) {
      selectionLabel.visible = false
    }
    return
  }

  // Create selection marker if needed (cyan ring, pulses in animate loop)
  if (!selectionMarker) {
    const geom = new THREE.RingGeometry(0.06, 0.09, 32)
    const mat = new THREE.MeshBasicMaterial({
      color: 0x00ddff,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false
    })
    selectionMarker = new THREE.Mesh(geom, mat)
    selectionMarker.renderOrder = 1000
    world.add(selectionMarker)
  }

  // Create selection label if needed
  if (!selectionLabel) {
    selectionLabel = createTextSprite('', '#1e90ff')
    selectionLabel.renderOrder = 1001
    world.add(selectionLabel)
  }

  // Update position and visibility
  selectionMarker.position.set(
    selectedObj.xyz_world[0],
    selectedObj.xyz_world[1],
    selectedObj.xyz_world[2]
  )
  // Make ring face camera
  selectionMarker.lookAt(camera.position)
  selectionMarker.visible = true

  // Update label text and position - short ID + stability (cyan to match ring)
  const labelText = `${selectedObj.id.slice(0, 8)} | ${selectedObj.stability.toFixed(2)}`
  const newLabel = createTextSprite(labelText, '#00ddff')
  // Make label always render on top
  ;(newLabel.material as THREE.SpriteMaterial).depthTest = false
  ;(newLabel.material as THREE.SpriteMaterial).depthWrite = false
  newLabel.position.set(
    selectedObj.xyz_world[0],
    selectedObj.xyz_world[1] + 0.12,
    selectedObj.xyz_world[2]
  )
  newLabel.renderOrder = 1001
  newLabel.visible = true

  // Replace old label
  if (selectionLabel) {
    world.remove(selectionLabel)
    ;(selectionLabel.material as THREE.SpriteMaterial).map?.dispose()
    ;(selectionLabel.material as THREE.Material).dispose()
  }
  world.add(newLabel)
  selectionLabel = newLabel
}

// Note: RTSM objects are now pushed via WebSocket (objects_update message)
// from the embedded visualization server. No polling needed.

// ============================================================================
// HUD
// ============================================================================

function updateHud() {
  const hudEl = document.getElementById('hud')
  if (!hudEl) return

  const confirmedCount = rtsmObjects.filter(o => o.confirmed).length
  hudEl.innerHTML = [
    `<b>RTSM Demo</b>`,
    `WS: ${wsConnected ? '<span style="color:#0f0">connected</span>' : '<span style="color:#f55">disconnected</span>'}`,
    `Meshes: ${meshes.size} | Points: ${totalPoints.toLocaleString()}`,
    `Pose updates: ${poseUpdateCount}`,
    `Objects: ${rtsmObjects.length} (${confirmedCount} confirmed)`,
  ].join('<br>')
}

// ============================================================================
// UI CONTROLS
// ============================================================================

const fileInput = document.getElementById('file') as HTMLInputElement
const loadBtn = document.getElementById('load') as HTMLButtonElement
const resetBtn = document.getElementById('reset') as HTMLButtonElement
const flipXBtn = document.getElementById('flipX') as HTMLButtonElement
const flipYBtn = document.getElementById('flipY') as HTMLButtonElement
const flipZBtn = document.getElementById('flipZ') as HTMLButtonElement
const toggleObjBtn = document.getElementById('toggleObj') as HTMLButtonElement
const filterConfirmedBtn = document.getElementById('filterConfirmed') as HTMLButtonElement
const clearStreamBtn = document.getElementById('clearStream') as HTMLButtonElement
const savePlyBtn = document.getElementById('savePly') as HTMLButtonElement
const rebuildBtn = document.getElementById('rebuild') as HTMLButtonElement
const disconnectBtn = document.getElementById('disconnect') as HTMLButtonElement
const reconnectBtn = document.getElementById('reconnect') as HTMLButtonElement
const modeEl = document.getElementById('mode') as HTMLSpanElement | null

// RTSM control elements
const rtsmResetBtn = document.getElementById('rtsmReset') as HTMLButtonElement
const rtsmStatsBtn = document.getElementById('rtsmStats') as HTMLButtonElement

// Object panel elements
const objectPanel = document.getElementById('object-panel')
const panelTitle = document.getElementById('panel-title')
const objectSearch = document.getElementById('object-search') as HTMLInputElement
const panelFilter = document.getElementById('panel-filter')
const panelToggle = document.getElementById('panel-toggle')
const objectListEl = document.getElementById('object-list')

let panelCollapsed = false
let searchQuery = ''
let selectedObjectId: string | null = null
let panelFilterConfirmed = false  // Filter to show only confirmed objects in panel

// Snapshot Gallery elements
const snapshotGallery = document.getElementById('snapshot-gallery')
const galleryClose = document.getElementById('gallery-close')
const galleryImages = document.getElementById('gallery-images')
const galleryPreview = document.getElementById('gallery-preview') as HTMLImageElement
const galleryInfo = document.getElementById('gallery-info')
const galleryLoading = document.getElementById('gallery-loading')
const galleryTitle = document.getElementById('gallery-title')
const ssRotateSelect = document.getElementById('ss-rotate') as HTMLSelectElement

// SS Rotate: apply CSS rotation to snapshot preview + thumbnails
ssRotateSelect?.addEventListener('change', () => {
  const rot = `rotate(${ssRotateSelect.value}deg)`
  if (galleryPreview) {
    galleryPreview.style.transform = rot
  }
  galleryImages?.querySelectorAll('.gallery-thumb').forEach((img) => {
    ;(img as HTMLElement).style.transform = rot
  })
})

// Gallery state
let gallerySnapshots: Array<{ index: number; data: string; size_bytes: number }> = []
let selectedSnapshotIndex = 0

function setMode(text: string) {
  if (modeEl) modeEl.textContent = `Mode: ${text}`
}
setMode('Free')

const initialCamPos = camera.position.clone()
const initialTarget = controls.target.clone()

let flippedX = false, flippedY = false, flippedZ = false

function applyFlipScale() {
  world.scale.set(flippedX ? -1 : 1, flippedY ? -1 : 1, flippedZ ? -1 : 1)
}

resetBtn?.addEventListener('click', () => {
  camera.position.copy(initialCamPos)
  camera.up.set(0, 1, 0)
  controls.target.copy(initialTarget)
  flippedX = flippedY = flippedZ = false
  applyFlipScale()
  controls.update()
  setMode('Free')
})

flipXBtn?.addEventListener('click', () => { flippedX = !flippedX; applyFlipScale() })
flipYBtn?.addEventListener('click', () => { flippedY = !flippedY; applyFlipScale() })
flipZBtn?.addEventListener('click', () => { flippedZ = !flippedZ; applyFlipScale() })

loadBtn?.addEventListener('click', () => {
  const f = fileInput?.files?.[0]
  if (f) loadPLYFile(f)
})

toggleObjBtn?.addEventListener('click', () => {
  showObjects = !showObjects
  updateObjectMarkers()
  toggleObjBtn.textContent = showObjects ? 'Hide Objects' : 'Show Objects'
})

if (filterConfirmedBtn) {
  filterConfirmedBtn.addEventListener('click', () => {
    showOnlyConfirmed = !showOnlyConfirmed
    console.log('[demo] Filter toggled, showOnlyConfirmed:', showOnlyConfirmed)
    updateObjectMarkers()
    // Button shows current filter state: "All" means showing all, "Confirmed Only" means filtered
    filterConfirmedBtn.textContent = showOnlyConfirmed ? 'Confirmed Only' : 'All'
  })
} else {
  console.warn('[demo] filterConfirmedBtn not found')
}

// Object panel toggle
panelToggle?.addEventListener('click', () => {
  panelCollapsed = !panelCollapsed
  objectPanel?.classList.toggle('collapsed', panelCollapsed)
  if (panelToggle) panelToggle.textContent = panelCollapsed ? '▲' : '▼'
})

// Object panel filter (All / Confirmed Only)
panelFilter?.addEventListener('click', () => {
  panelFilterConfirmed = !panelFilterConfirmed
  panelFilter.textContent = panelFilterConfirmed ? 'Confirmed' : 'All'
  panelFilter.classList.toggle('active', panelFilterConfirmed)
  updateObjectList()
})

// Semantic search button and elements
const searchBtn = document.getElementById('search-btn')
const showAllBtn = document.getElementById('show-all-btn')
const searchThresholdInput = document.getElementById('search-threshold') as HTMLInputElement

// Perform search: UNION of detector-label search and semantic (CLIP +
// FAISS) search, label hits ranked first, deduped — mirrors the agent's
// retrieval policy (2026-08-28). Label search reaches PROTOS (objects
// the detector sees right now that semantic search can't surface until
// confirmation + indexing), so the viz shows what the robot can act on.
async function performSemanticSearch() {
  const query = objectSearch?.value.trim()
  if (!query) {
    console.log('[search] Empty query, showing all objects')
    showAllObjects()
    return
  }

  const threshold = parseFloat(searchThresholdInput?.value || '0.0') || 0.0
  searchQuery = query
  console.log(`[search] Searching for: "${query}" (threshold=${threshold})`)

  try {
    const [labelRes, semRes] = await Promise.allSettled([
      fetch(`${RTSM_API_BASE}/search/label?query=${encodeURIComponent(query)}&top_k=10`),
      fetch(`${RTSM_API_BASE}/search/semantic?query=${encodeURIComponent(query)}&top_k=10&threshold=${threshold}`),
    ])

    semanticSearchResults.clear()
    let labelCount = 0
    if (labelRes.status === 'fulfilled' && labelRes.value.ok) {
      const data = await labelRes.value.json()
      console.log('[search] label API response:', data)
      for (const r of data.results ?? []) {
        semanticSearchResults.set(r.id, r.score)
        labelCount++
      }
    }
    if (semRes.status === 'fulfilled' && semRes.value.ok) {
      const data = await semRes.value.json()
      console.log('[search] semantic API response:', data)
      for (const r of data.results ?? []) {
        if (!semanticSearchResults.has(r.id)) {
          semanticSearchResults.set(r.id, r.score)
        }
      }
    } else if (labelCount === 0) {
      throw new Error('both search endpoints failed')
    }
    isSearchMode = true
    updateObjectList()
    updateObjectMarkers()  // Update 3D view to show only matches
    console.log(`[search] ${semanticSearchResults.size} matches (${labelCount} by label)`)
  } catch (e) {
    console.error('[search] API call failed:', e)
    alert('Search failed. Is RTSM running?')
  }
}

// Show all objects (clear search mode)
function showAllObjects() {
  searchQuery = ''
  semanticSearchResults.clear()
  isSearchMode = false
  if (objectSearch) objectSearch.value = ''
  updateObjectList()
  updateObjectMarkers()  // Update 3D view to show all objects again
}

// Search button click
searchBtn?.addEventListener('click', performSemanticSearch)

// Show All button click
showAllBtn?.addEventListener('click', showAllObjects)

// Enter key in search input triggers search
objectSearch?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault()
    performSemanticSearch()
  }
})

// Update object list in panel
function updateObjectList() {
  if (!objectListEl) return

  // Filter objects based on semantic search results or show all
  let filtered: RtsmObject[]
  if (isSearchMode) {
    // Semantic search mode: only show objects in search results
    filtered = rtsmObjects.filter(obj => {
      if (panelFilterConfirmed && !obj.confirmed) return false
      return semanticSearchResults.has(obj.id)
    })
    // Sort by similarity score (highest first)
    filtered.sort((a, b) => {
      const scoreA = semanticSearchResults.get(a.id) || 0
      const scoreB = semanticSearchResults.get(b.id) || 0
      return scoreB - scoreA
    })
  } else {
    // Show all objects
    filtered = rtsmObjects.filter(obj => {
      if (panelFilterConfirmed && !obj.confirmed) return false
      return true
    })
  }

  // Update title count
  if (panelTitle) {
    const confirmedCount = filtered.filter(o => o.confirmed).length
    if (isSearchMode) {
      panelTitle.textContent = `Search "${searchQuery}": ${filtered.length} matches`
    } else {
      panelTitle.textContent = `Objects (${filtered.length}) — ${confirmedCount} confirmed`
    }
  }

  objectListEl.innerHTML = filtered.map(obj => {
    const isSelected = obj.id === selectedObjectId
    const semanticScore = semanticSearchResults.get(obj.id)

    // In search mode show similarity, otherwise stability
    const scoreStr = isSearchMode && semanticScore !== undefined
      ? `sim ${semanticScore.toFixed(2)}`
      : `${obj.stability.toFixed(2)}`

    const statusTag = obj.confirmed ? 'confirmed' : 'proto'

    return `
      <div class="object-item ${isSelected ? 'selected' : ''}" data-id="${obj.id}">
        <div class="object-dot ${statusTag}"></div>
        <div class="object-info">
          <div class="object-label">${obj.id.slice(0, 8)}</div>
          <div class="object-meta">${statusTag} &middot; ${scoreStr}</div>
        </div>
      </div>
    `
  }).join('')

  // Add click handlers
  objectListEl.querySelectorAll('.object-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = item.getAttribute('data-id')
      if (id) selectObject(id)
    })
  })
}

// Select object (click again to deselect) - 3D view stays the same
function selectObject(id: string) {
  // Toggle selection if clicking on already selected object
  if (selectedObjectId === id) {
    selectedObjectId = null
    snapshotGallery?.classList.remove('visible')  // Close gallery on deselect
  } else {
    selectedObjectId = id
    loadObjectSnapshots(id)  // Load snapshots for selected object
  }
  updateObjectList()
  updateObjectMarkers() // Update selection marker
}

// ============================================================================
// SNAPSHOT GALLERY
// ============================================================================

// Gallery close handler
galleryClose?.addEventListener('click', () => {
  snapshotGallery?.classList.remove('visible')
})

// Make snapshot gallery draggable by its header
{
  const header = document.getElementById('gallery-header')
  const panel = snapshotGallery
  if (header && panel) {
    let isDragging = false
    let startX = 0, startY = 0, startLeft = 0, startTop = 0

    header.addEventListener('mousedown', (e: MouseEvent) => {
      if ((e.target as HTMLElement).tagName === 'BUTTON' || (e.target as HTMLElement).tagName === 'SELECT') return
      isDragging = true
      const rect = panel.getBoundingClientRect()
      startX = e.clientX
      startY = e.clientY
      startLeft = rect.left
      startTop = rect.top
      // Switch from fixed bottom/right to fixed top/left positioning
      panel.style.left = rect.left + 'px'
      panel.style.top = rect.top + 'px'
      panel.style.right = 'auto'
      panel.style.bottom = 'auto'
      e.preventDefault()
    })

    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (!isDragging) return
      const dx = e.clientX - startX
      const dy = e.clientY - startY
      panel.style.left = (startLeft + dx) + 'px'
      panel.style.top = (startTop + dy) + 'px'
    })

    document.addEventListener('mouseup', () => {
      isDragging = false
    })
  }
}

// Load snapshots for an object
async function loadObjectSnapshots(objectId: string) {
  if (!snapshotGallery || !galleryImages || !galleryLoading) return

  snapshotGallery.classList.add('visible')
  galleryImages.innerHTML = ''
  galleryPreview?.classList.remove('visible')
  if (galleryLoading) {
    galleryLoading.style.display = 'block'
    galleryLoading.textContent = 'Loading snapshots...'
  }

  if (galleryTitle) {
    galleryTitle.textContent = `Snapshots: ${objectId.slice(0, 8)}`
  }

  try {
    const resp = await fetch(`${RTSM_API_BASE}/objects/${objectId}/snapshots`)
    const data = await resp.json()

    if (data.error) {
      if (galleryLoading) galleryLoading.textContent = `Error: ${data.error}`
      return
    }

    gallerySnapshots = data.snapshots || []

    if (gallerySnapshots.length === 0) {
      if (galleryLoading) galleryLoading.textContent = 'No snapshots available'
      return
    }

    if (galleryLoading) galleryLoading.style.display = 'none'

    // Create thumbnails
    gallerySnapshots.forEach((snap, i) => {
      const img = document.createElement('img')
      img.className = 'gallery-thumb' + (i === 0 ? ' selected' : '')
      img.src = snap.data
      if (ssRotateSelect?.value && ssRotateSelect.value !== '0') {
        img.style.transform = `rotate(${ssRotateSelect.value}deg)`
      }
      img.alt = `Snapshot ${i + 1}`
      img.dataset.index = String(i)
      img.addEventListener('click', () => selectSnapshot(i))
      galleryImages.appendChild(img)
    })

    // Show first snapshot
    selectSnapshot(0)

    if (galleryInfo) {
      galleryInfo.textContent = `${gallerySnapshots.length} snapshot(s) | Most recent first`
    }

  } catch (e) {
    console.error('[gallery] Failed to load snapshots:', e)
    if (galleryLoading) {
      galleryLoading.style.display = 'block'
      galleryLoading.textContent = 'Failed to load snapshots'
    }
  }
}

function selectSnapshot(index: number) {
  if (!galleryImages || !galleryPreview) return

  selectedSnapshotIndex = index

  // Update thumbnail selection
  galleryImages.querySelectorAll('.gallery-thumb').forEach((img, i) => {
    img.classList.toggle('selected', i === index)
  })

  // Show preview
  const snap = gallerySnapshots[index]
  if (snap) {
    galleryPreview.src = snap.data
    galleryPreview.classList.add('visible')
  }
}

clearStreamBtn?.addEventListener('click', () => {
  // Send clear command to server
  sendCommand('clear')
  // Also clear locally
  clearAllMeshes()
})

rebuildBtn?.addEventListener('click', () => {
  // Request rebuild from cache - clear local and reconnect to get fresh sync
  clearAllMeshes()
  if (ws) {
    ws.close()
  }
})

// Helper to update connection UI (tab bar status dot + button states)
const connStatusDot = document.getElementById('conn-status')

function updateConnectionButtons() {
  if (disconnectBtn) disconnectBtn.disabled = !wsConnected
  if (reconnectBtn) reconnectBtn.disabled = wsConnected
  // Update status dot in tab bar
  if (connStatusDot) {
    connStatusDot.classList.toggle('connected', wsConnected)
    connStatusDot.classList.toggle('disconnected', !wsConnected)
    connStatusDot.title = wsConnected ? 'Connected' : 'Disconnected'
  }
}

// Disconnect button - clears view and disconnects WebSocket (stays disconnected)
disconnectBtn?.addEventListener('click', () => {
  // Prevent auto-reconnect
  isManuallyDisconnected = true

  // Clear all meshes and objects from view
  clearAllMeshes()
  rtsmObjects = []
  updateObjectMarkers()
  updateObjectList()

  // Disconnect WebSocket
  if (ws) {
    ws.close()
    ws = null
  }

  updateConnectionButtons()
})

// Re-connect button - reconnects immediately (no delay)
reconnectBtn?.addEventListener('click', () => {
  // Allow auto-reconnect again
  isManuallyDisconnected = false

  // Connect immediately
  connectWebSocket()
})

// ============================================================================
// RTSM API CONTROLS
// ============================================================================

// In dev: Vite proxies /api -> RTSM API (strips prefix). In production: served
// from same origin, so use empty prefix (routes are at /objects, /search, etc.)
const RTSM_API_BASE = import.meta.env.DEV ? '/api' : ''

// Reset RTSM (clears WM, sweep cache, frame window, visualization)
rtsmResetBtn?.addEventListener('click', async () => {
  if (!confirm('Reset RTSM? This will clear all objects, keyframes, and sweep state.')) {
    return
  }
  rtsmResetBtn.disabled = true
  rtsmResetBtn.textContent = 'Resetting...'
  try {
    const resp = await fetch(`${RTSM_API_BASE}/reset`, { method: 'POST' })
    const data = await resp.json()
    console.log('[RTSM] Reset result:', data)

    // Also clear local visualization
    clearAllMeshes()
    rtsmObjects = []
    updateObjectMarkers()
    updateObjectList()
    updateHud()

    alert(`RTSM Reset Complete!\n\nCleared:\n- ${data.cleared?.working_memory?.objects_cleared || 0} objects\n- ${data.cleared?.sweep_cache?.view_states_cleared || 0} sweep states\n- ${data.cleared?.visualization?.keyframes_cleared || 0} keyframes`)
  } catch (e) {
    console.error('[RTSM] Reset failed:', e)
    alert('Failed to reset RTSM. Is the API server running?')
  } finally {
    rtsmResetBtn.disabled = false
    rtsmResetBtn.textContent = 'Reset WM'
  }
})

// Show RTSM stats
rtsmStatsBtn?.addEventListener('click', async () => {
  rtsmStatsBtn.disabled = true
  rtsmStatsBtn.textContent = 'Loading...'
  try {
    const resp = await fetch(`${RTSM_API_BASE}/stats/detailed`)
    const data = await resp.json()
    console.log('[RTSM] Stats:', data)

    const wm = data.working_memory || {}
    const sc = data.sweep_cache || {}
    const fw = data.frame_window || {}
    const vis = data.visualization || {}

    alert([
      'RTSM Statistics',
      '═══════════════════════',
      '',
      'Working Memory:',
      `  Objects: ${wm.objects || 0} (${wm.confirmed || 0} confirmed)`,
      `  Avg Hits: ${(wm.avg_hits || 0).toFixed(1)}`,
      `  Upserts: ${wm.upserts_total || 0}`,
      '',
      'Sweep Cache:',
      `  Cells: ${sc.cells || 0}`,
      `  View States: ${sc.view_states || 0}`,
      `  Cam Snapshots: ${sc.cam_snapshots || 0}`,
      '',
      'Frame Buffer:',
      `  RGB Frames: ${fw.rgb_frames || 0}`,
      `  Depth Frames: ${fw.depth_frames || 0}`,
      '',
      'Visualization:',
      `  Keyframes: ${vis.keyframes || 0}`,
      `  Total Points: ${(vis.total_points || 0).toLocaleString()}`,
    ].join('\n'))
  } catch (e) {
    console.error('[RTSM] Stats fetch failed:', e)
    alert('Failed to fetch RTSM stats. Is the API server running?')
  } finally {
    rtsmStatsBtn.disabled = false
    rtsmStatsBtn.textContent = 'Stats'
  }
})

// Export PLY
function gatherWorldPoints(): { positions: Float32Array; colors: Float32Array } {
  const posOut: number[] = []
  const colOut: number[] = []
  const m = new THREE.Matrix4()
  const v = new THREE.Vector3()

  for (const pts of meshes.values()) {
    const posAttr = pts.geometry.getAttribute('position')
    const colAttr = pts.geometry.getAttribute('color')
    if (!posAttr) continue
    m.copy(pts.matrix)
    for (let i = 0; i < posAttr.count; i++) {
      v.fromBufferAttribute(posAttr, i)
      v.applyMatrix4(m)
      posOut.push(v.x, v.y, v.z)
      if (colAttr) {
        colOut.push(colAttr.getX(i), colAttr.getY(i), colAttr.getZ(i))
      } else {
        colOut.push(1, 1, 1)
      }
    }
  }

  // Also include static PLY if loaded
  if (currentPoints) {
    const posAttr = currentPoints.geometry.getAttribute('position')
    const colAttr = currentPoints.geometry.getAttribute('color')
    if (posAttr) {
      for (let i = 0; i < posAttr.count; i++) {
        v.fromBufferAttribute(posAttr, i)
        posOut.push(v.x, v.y, v.z)
        if (colAttr) {
          colOut.push(colAttr.getX(i), colAttr.getY(i), colAttr.getZ(i))
        } else {
          colOut.push(1, 1, 1)
        }
      }
    }
  }

  return {
    positions: new Float32Array(posOut),
    colors: new Float32Array(colOut)
  }
}

function exportPLY(positions: Float32Array, colors: Float32Array): string {
  const n = positions.length / 3
  const header = [
    'ply',
    'format ascii 1.0',
    `element vertex ${n}`,
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
  ].join('\n')
  const body: string[] = []
  for (let i = 0; i < n; i++) {
    const r = Math.round(colors[i * 3] * 255)
    const g = Math.round(colors[i * 3 + 1] * 255)
    const b = Math.round(colors[i * 3 + 2] * 255)
    body.push(`${positions[i * 3]} ${positions[i * 3 + 1]} ${positions[i * 3 + 2]} ${r} ${g} ${b}`)
  }
  return header + '\n' + body.join('\n') + '\n'
}

savePlyBtn?.addEventListener('click', () => {
  const { positions, colors } = gatherWorldPoints()
  if (positions.length === 0) {
    console.log('[demo] No points to export')
    return
  }
  const ply = exportPLY(positions, colors)
  const blob = new Blob([ply], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'rtsm_pointcloud.ply'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
  console.log(`[demo] Exported ${positions.length / 3} points`)
})

// ============================================================================
// KEYBOARD CONTROLS (axis-constrained rotation)
// ============================================================================

let pitchTemp = false, yawTemp = false, rollTemp = false
let rollLastX = 0, lastPointerX = 0
let isZDown = false, isXDown = false, isCDown = false
let isDragging = false

function setPitchOnly(enabled: boolean) {
  if (enabled) {
    const az = controls.getAzimuthalAngle()
    controls.minAzimuthAngle = az - 1e-4
    controls.maxAzimuthAngle = az + 1e-4
  } else {
    controls.minAzimuthAngle = -Infinity
    controls.maxAzimuthAngle = Infinity
  }
}

function setYawOnly(enabled: boolean) {
  if (enabled) {
    const pol = controls.getPolarAngle()
    controls.minPolarAngle = Math.max(0, pol - 1e-4)
    controls.maxPolarAngle = Math.min(Math.PI, pol + 1e-4)
  } else {
    controls.minPolarAngle = 0.0001
    controls.maxPolarAngle = Math.PI - 0.0001
  }
}

function applyModeLocks() {
  if (rollTemp) {
    setMode('Roll-only')
    return
  }
  if (pitchTemp && !yawTemp) {
    setYawOnly(false)
    setPitchOnly(true)
    setMode('Pitch-only')
  } else if (yawTemp && !pitchTemp) {
    setPitchOnly(false)
    setYawOnly(true)
    setMode('Yaw-only')
  } else {
    setPitchOnly(false)
    setYawOnly(false)
    setMode('Free')
  }
}

function clearTempLocks() {
  pitchTemp = false
  yawTemp = false
  applyModeLocks()
}

controls.addEventListener('start', () => { isDragging = true })
controls.addEventListener('end', () => { isDragging = false; clearTempLocks() })

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return
  lastPointerX = e.clientX
  if (isCDown) {
    rollTemp = true
    rollLastX = e.clientX
    ;(controls as any).enableRotate = false
    applyModeLocks()
    e.preventDefault()
  } else if (isXDown) {
    yawTemp = true
    applyModeLocks()
  } else if (isZDown) {
    pitchTemp = true
    applyModeLocks()
  }
}, { capture: true })

renderer.domElement.addEventListener('pointerup', (e) => {
  if (e.button !== 0) return
  if (rollTemp) {
    rollTemp = false
    ;(controls as any).enableRotate = true
    applyModeLocks()
  } else {
    clearTempLocks()
  }
}, { capture: true })

renderer.domElement.addEventListener('pointermove', (e) => {
  lastPointerX = e.clientX
  if (!rollTemp) return
  const dx = e.clientX - rollLastX
  if (dx !== 0) {
    const viewAxis = new THREE.Vector3().subVectors(controls.target, camera.position).normalize()
    const q = new THREE.Quaternion().setFromAxisAngle(viewAxis, dx * 0.005)
    camera.up.applyQuaternion(q).normalize()
    camera.lookAt(controls.target)
    rollLastX = e.clientX
  }
  e.preventDefault()
}, { capture: true })

document.addEventListener('keydown', (e) => {
  if (e.key === 'z' || e.key === 'Z') isZDown = true
  if (e.key === 'x' || e.key === 'X') isXDown = true
  if (e.key === 'c' || e.key === 'C') isCDown = true

  if (!isDragging) return
  if (e.key === 'c' || e.key === 'C') {
    if (!rollTemp) {
      rollTemp = true
      rollLastX = lastPointerX
      ;(controls as any).enableRotate = false
      applyModeLocks()
    }
  } else if (e.key === 'x' || e.key === 'X') {
    yawTemp = true
    applyModeLocks()
  } else if (e.key === 'z' || e.key === 'Z') {
    pitchTemp = true
    applyModeLocks()
  }
})

document.addEventListener('keyup', (e) => {
  if (e.key === 'z' || e.key === 'Z') isZDown = false
  if (e.key === 'x' || e.key === 'X') isXDown = false
  if (e.key === 'c' || e.key === 'C') isCDown = false

  if (!isDragging) return
  if ((e.key === 'c' || e.key === 'C') && rollTemp) {
    rollTemp = false
    ;(controls as any).enableRotate = true
    applyModeLocks()
  } else if (e.key === 'x' || e.key === 'X' || e.key === 'z' || e.key === 'Z') {
    clearTempLocks()
  }
})

// Double-click to set orbit focus
const raycaster = new THREE.Raycaster()
const mouse = new THREE.Vector2()

renderer.domElement.addEventListener('dblclick', (event) => {
  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
  raycaster.setFromCamera(mouse, camera)

  // Check static cloud
  if (currentPoints) {
    raycaster.params.Points.threshold = 0.01
    const hits = raycaster.intersectObject(currentPoints, false)
    if (hits.length > 0) {
      controls.target.copy(hits[0].point)
      controls.update()
      return
    }
  }

  // Check streamed clouds
  for (const pts of meshes.values()) {
    raycaster.params.Points.threshold = 0.01
    const hits = raycaster.intersectObject(pts, false)
    if (hits.length > 0) {
      controls.target.copy(hits[0].point)
      controls.update()
      return
    }
  }
})

// ============================================================================
// ANIMATION LOOP
// ============================================================================

function animate() {
  requestAnimationFrame(animate)

  // Pulse selected object marker (cyan blink, 2Hz sine wave)
  if (selectionMarker && selectionMarker.visible) {
    const t = performance.now() / 1000
    const pulse = 0.5 + 0.5 * Math.sin(t * Math.PI * 4) // 2Hz, range 0..1
    const mat = selectionMarker.material as THREE.MeshBasicMaterial
    mat.opacity = 0.4 + pulse * 0.6  // range 0.4..1.0
    // Also scale-pulse the marker subtly
    const s = 1.0 + pulse * 0.15  // range 1.0..1.15
    selectionMarker.scale.set(s, s, s)
  }

  // Pulse the selected object's sphere marker too
  if (selectedObjectId) {
    const marker = objectMarkers.get(selectedObjectId)
    if (marker) {
      const t = performance.now() / 1000
      const pulse = 0.5 + 0.5 * Math.sin(t * Math.PI * 4)
      ;(marker.material as THREE.MeshBasicMaterial).color.setHex(
        pulse > 0.5 ? 0x00ddff : 0x1e90ff  // blink between cyan and blue
      )
      const s = 1.5 + pulse * 0.5  // range 1.5..2.0
      marker.scale.set(s, s, s)
    }
  }

  renderer.render(scene, camera)
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})

// ============================================================================
// RUNTIME ANALYTICS DASHBOARD
// ============================================================================

let analyticsVisible = false
let latencyHistory: any[] = []
let segHistory: any[] = []
let latestAggregate: { latency?: any; segmentation?: any } = {}
let runtimeConfig: any = null
let throughputChart: uPlot | null = null
let latencyChart: uPlot | null = null
let segChart: uPlot | null = null
let configCollapsed = false

const configToggle = document.getElementById('analytics-config-toggle')

function pruneByTime(arr: any[], maxAge: number) {
  const cutoff = Date.now() / 1000 - maxAge
  while (arr.length > 0 && arr[0].wall_ts < cutoff) arr.shift()
}

function handleAnalyticsMessage(msg: any) {
  if (msg.mode === 'full') {
    latencyHistory = msg.latency?.hourly ?? []
    segHistory = msg.segmentation?.hourly ?? []
    if (msg.config) {
      runtimeConfig = msg.config
      if (analyticsVisible) {
        renderConfigPanel()
        adaptSegmentationSection()
      }
    }
  } else if (msg.mode === 'append') {
    if (msg.latency?.bucket) {
      latencyHistory.push(msg.latency.bucket)
      pruneByTime(latencyHistory, 3600)
    }
    if (msg.segmentation?.bucket) {
      segHistory.push(msg.segmentation.bucket)
      pruneByTime(segHistory, 3600)
    }
  }
  latestAggregate = {
    latency: msg.latency?.aggregate,
    segmentation: msg.segmentation?.aggregate,
  }
  if (analyticsVisible) updateAnalyticsPanel()
}

// ---- Tab switching (event delegation for reliability) ----

function switchTab(tab: string) {
  document.querySelectorAll('#tab-bar .tab-btn').forEach(t =>
    t.classList.toggle('active', (t as HTMLElement).dataset.tab === tab)
  )
  const v3d = document.getElementById('view-3d')
  const vAn = document.getElementById('view-analytics')

  if (tab === '3d') {
    if (v3d) v3d.style.display = 'block'
    if (vAn) vAn.style.display = 'none'
    analyticsVisible = false
  } else {
    if (v3d) v3d.style.display = 'none'
    if (vAn) vAn.style.display = 'block'
    analyticsVisible = true
    initChartsIfNeeded()
    if (runtimeConfig) {
      renderConfigPanel()
      adaptSegmentationSection()
    }
    updateAnalyticsPanel()
  }
}

document.getElementById('tab-bar')?.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest('.tab-btn') as HTMLElement | null
  if (btn?.dataset.tab) switchTab(btn.dataset.tab)
})

configToggle?.addEventListener('click', () => {
  configCollapsed = !configCollapsed
  const el = document.getElementById('analytics-config')
  if (el) el.style.display = configCollapsed ? 'none' : ''
  if (configToggle) configToggle.textContent = configCollapsed ? 'Config \u25b8' : 'Config \u25be'
})

// ---- Config panel ----

function renderConfigPanel() {
  const el = document.getElementById('analytics-config')
  if (!el || !runtimeConfig) return
  const seg = runtimeConfig.segmentation || {}
  const rx = runtimeConfig.receiver || {}
  const pl = runtimeConfig.pipeline || {}
  const fsam = seg.fastsam || {}
  const yoloe = seg.yoloe || {}

  const item = (k: string, v: any, hl?: boolean) =>
    `<span class="config-item"><span class="config-key">${k}:</span> <span class="${hl ? 'config-highlight' : 'config-val'}">${v}</span></span>`

  let items = [
    item('backend', seg.backend, true),
    item('device', pl.device ?? '?'),
    item('retina', seg.retina_masks ?? '?'),
  ]

  if (seg.backend === 'dual') {
    items.push(
      item('fastsam', `${fsam.imgsz}px conf=${fsam.conf} iou=${fsam.iou}`),
      item('yoloe', `${yoloe.imgsz}px conf=${yoloe.conf} iou=${yoloe.iou}`),
      item('dual-iou', seg.dual?.iou_confirm_threshold),
      item('prefer', seg.dual?.prefer_mask),
    )
  } else {
    const a = seg[seg.backend] || {}
    items.push(item('imgsz', `${a.imgsz}px`), item('conf', a.conf), item('max-det', a.max_det))
  }

  items.push(
    item('kf-every', `${rx.keyframe_every_n}f`),
    item('throttle', `${rx.nonkf_min_interval_s}s`),
    item('queue', rx.queue_maxsize),
    item('top-k', pl.topk_preclip),
  )

  el.innerHTML = `<div class="config-grid">${items.join('')}</div>`
}

// ---- Adaptive segmentation section ----

function adaptSegmentationSection() {
  const label = document.querySelector('#analytics-seg-section .chart-section-label')
  if (!label || !runtimeConfig) return
  const backend = runtimeConfig.segmentation?.backend
  if (backend === 'dual') {
    label.textContent = 'Segmentation (Dual Confirmation)'
  } else {
    label.textContent = `Segmentation (${backend} only)`
  }
}

// ---- Stats rendering ----

function updateAnalyticsPanel() {
  updateKPIs()
  updateLatencyBreakdown()
  updateChartDetails()
  updateCharts()
}

// ---- KPI scorecard row ----

function setKPI(id: string, value: string, cls?: string) {
  const card = document.getElementById(id)
  if (!card) return
  const valEl = card.querySelector('.kpi-value')
  if (valEl) {
    valEl.textContent = value
    valEl.className = 'kpi-value' + (cls ? ' ' + cls : '')
  }
}

function setKPISub(id: string, text: string) {
  const card = document.getElementById(id)
  if (!card) return
  const subEl = card.querySelector('.kpi-sub')
  if (subEl) subEl.textContent = text
}

function updateKPIs() {
  const la = latestAggregate.latency
  const sa = latestAggregate.segmentation
  const lastB = latencyHistory.length > 0 ? latencyHistory[latencyHistory.length - 1] : null

  if (la) {
    setKPI('kpi-input-hz', `${la.input_hz ?? 0}`)
    setKPI('kpi-proc-hz', `${la.processing_hz ?? 0}`)
    setKPISub('kpi-proc-hz', `Hz (${((la.effective_ratio ?? 0) * 100).toFixed(0)}% ratio)`)

    const totalMs = la.t_total?.mean ? (la.t_total.mean * 1000).toFixed(0) : '—'
    const p95Ms = la.t_total?.p95 ? (la.t_total.p95 * 1000).toFixed(0) : '?'
    const ltCls = Number(totalMs) > 500 ? 'yellow' : Number(totalMs) > 1000 ? 'red' : ''
    setKPI('kpi-latency', totalMs, ltCls)
    setKPISub('kpi-latency', `ms mean (p95: ${p95Ms}ms)`)

    const qMax = lastB?.queue_depth_max ?? 0
    const qDrops = lastB?.queue_drops ?? 0
    const qCls = qDrops > 0 ? 'red' : qMax > 400 ? 'red' : qMax > 256 ? 'yellow' : 'green'
    setKPI('kpi-queue', `${qMax}`, qCls)
    setKPISub('kpi-queue', qDrops > 0 ? `/ 512 cap  (${qDrops} drops!)` : '/ 512 cap')
    const qCard = document.getElementById('kpi-queue')
    if (qCard) qCard.classList.toggle('congestion', qDrops > 0)

    const surv = ((la.mask_survival_rate ?? 0) * 100).toFixed(0)
    setKPI('kpi-masks', `${la.mean_masks_in ?? 0}`)
    setKPISub('kpi-masks', `\u2192 ${la.mean_candidates ?? 0} selected (${surv}%)`)
  }

  if (sa) {
    const backend = sa.backend ?? runtimeConfig?.segmentation?.backend ?? '?'
    if (backend === 'dual') {
      setKPI('kpi-dual', `${((sa.dual_rate ?? 0) * 100).toFixed(0)}%`, 'green')
      setKPISub('kpi-dual', `dual | ${((sa.fastsam_only_rate ?? 0) * 100).toFixed(0)}% fsam | ${((sa.yoloe_only_rate ?? 0) * 100).toFixed(0)}% yoloe`)
    } else {
      setKPI('kpi-dual', `${sa.mean_total ?? 0}`)
      setKPISub('kpi-dual', `masks/frame (${backend})`)
    }
  }

  // Objects — use latest non-empty bucket's WM snapshot (always reflects current state)
  // Association — sum across all buckets for session totals
  if (latencyHistory.length > 0) {
    // Find latest bucket with WM data (wm_total > 0 or any frames)
    let wmBucket = lastB
    for (let i = latencyHistory.length - 1; i >= 0; i--) {
      if ((latencyHistory[i].wm_total ?? 0) > 0 || (latencyHistory[i].frames_in_bucket ?? 0) > 0) {
        wmBucket = latencyHistory[i]
        break
      }
    }

    const confirmed = wmBucket?.wm_confirmed ?? 0
    const total = wmBucket?.wm_total ?? 0
    const proto = wmBucket?.wm_proto ?? 0
    const objCls = confirmed > 0 ? 'green' : total > 0 ? 'yellow' : ''
    setKPI('kpi-objects', `${confirmed}`, objCls)
    setKPISub('kpi-objects', `confirmed / ${total} total (${proto} proto)`)

    // Sum association across all buckets for session totals
    let totalMatched = 0, totalCreated = 0
    for (const b of latencyHistory) {
      totalMatched += b.assoc_matched ?? 0
      totalCreated += b.assoc_created ?? 0
    }
    const matchRate = (totalMatched + totalCreated) > 0
      ? ((totalMatched / (totalMatched + totalCreated)) * 100).toFixed(0) : '—'
    setKPI('kpi-assoc', `${totalMatched}`, totalMatched > 0 ? 'green' : '')
    setKPISub('kpi-assoc', `matched / ${totalCreated} new (${matchRate}% match rate)`)
  }
}

// ---- Latency breakdown (inline colored dots above chart) ----

function updateLatencyBreakdown() {
  const el = document.getElementById('analytics-latency-breakdown')
  if (!el) return
  const a = latestAggregate.latency
  if (!a) { el.innerHTML = ''; return }

  const totalMean = a.t_total?.mean ?? 0
  const stages = [
    { name: 'Seg', stats: a.t_segmentation, color: '#ff6b6b' },
    { name: 'Heur', stats: a.t_heuristics, color: '#ffd93d' },
    { name: 'Score', stats: a.t_scoring, color: '#aaaaaa' },
    { name: 'CLIP', stats: a.t_clip, color: '#6bcb77' },
    { name: 'Assoc', stats: a.t_association, color: '#4d96ff' },
  ]

  el.innerHTML = '<div class="latency-breakdown">' + stages.map(s => {
    const mean = s.stats?.mean ?? 0
    const ms = (mean * 1000).toFixed(0)
    const pct = totalMean > 0 ? ((mean / totalMean) * 100).toFixed(0) : '0'
    return `<div class="latency-item">
      <span class="latency-dot" style="background:${s.color}"></span>
      <span class="latency-name">${s.name}</span>
      <span class="latency-val">${ms}ms</span>
      <span class="latency-pct">(${pct}%)</span>
    </div>`
  }).join('') + '</div>'
}

// ---- Chart section detail text (inline next to label) ----

function updateChartDetails() {
  const la = latestAggregate.latency
  const sa = latestAggregate.segmentation
  const lastB = latencyHistory.length > 0 ? latencyHistory[latencyHistory.length - 1] : null

  const tpDetail = document.getElementById('analytics-throughput-detail')
  if (tpDetail && la) {
    const throttle = lastB?.throttle_skips ?? 0
    const gate = lastB?.gate_rejections ?? 0
    tpDetail.textContent = `throttle: ${throttle}/s | gate: ${gate}/s`
  }

  const ltDetail = document.getElementById('analytics-latency-detail')
  if (ltDetail && la) {
    const p95 = la.t_total?.p95 ? (la.t_total.p95 * 1000).toFixed(0) : '?'
    const max = la.t_total?.max ? (la.t_total.max * 1000).toFixed(0) : '?'
    ltDetail.textContent = `p95: ${p95}ms | max: ${max}ms`
  }

  const segDetail = document.getElementById('analytics-seg-detail')
  if (segDetail && sa) {
    const backend = sa.backend ?? runtimeConfig?.segmentation?.backend ?? '?'
    if (backend === 'dual') {
      segDetail.textContent = `raw: FastSAM ${sa.mean_fastsam_raw ?? 0} | YOLOE ${sa.mean_yoloe_raw ?? 0} | survival: ${((sa.staged_survival_rate ?? 0) * 100).toFixed(0)}%`
    } else {
      segDetail.textContent = `${sa.mean_total ?? 0} masks/frame | survival: ${((sa.staged_survival_rate ?? 0) * 100).toFixed(0)}%`
    }
  }
}

// ---- uPlot charts ----

const DARK_AXES: uPlot.Axis = { stroke: '#555', grid: { stroke: '#222' } }
const DARK_AXES_Y: uPlot.Axis = { stroke: '#555', grid: { stroke: '#222' }, size: 45 }

function chartWidth(containerId: string): number {
  const el = document.getElementById(containerId)
  return el ? Math.max(300, el.clientWidth) : 600
}

function initChartsIfNeeded() {
  if (!throughputChart) {
    const el = document.getElementById('analytics-throughput-chart')
    if (el) {
      throughputChart = new uPlot({
        width: chartWidth('analytics-throughput-chart'), height: 150,
        series: [
          {},
          { label: 'Input Hz', stroke: '#888', width: 1 },
          { label: 'Proc Hz', stroke: '#1e90ff', width: 2 },
          { label: 'Q Max', stroke: '#ffffff40', fill: '#ffffff10', width: 1 },
        ],
        axes: [DARK_AXES, DARK_AXES_Y],
        cursor: { show: true },
        legend: { show: false },
      }, [[], [], [], []], el)
    }
  }
  if (!latencyChart) {
    const el = document.getElementById('analytics-latency-chart')
    if (el) {
      latencyChart = new uPlot({
        width: chartWidth('analytics-latency-chart'), height: 150,
        series: [
          {},
          { label: 'Seg', stroke: '#ff6b6b', fill: '#ff6b6b40', width: 1 },
          { label: 'Heur', stroke: '#ffd93d', fill: '#ffd93d40', width: 1 },
          { label: 'Score', stroke: '#aaa', fill: '#aaaaaa30', width: 1 },
          { label: 'CLIP', stroke: '#6bcb77', fill: '#6bcb7740', width: 1 },
          { label: 'Assoc', stroke: '#4d96ff', fill: '#4d96ff40', width: 1 },
        ],
        axes: [DARK_AXES, DARK_AXES_Y],
        cursor: { show: true },
        legend: { show: false },
      }, [[], [], [], [], [], []], el)
    }
  }
  if (!segChart) {
    const el = document.getElementById('analytics-seg-chart')
    if (el) {
      const isDual = runtimeConfig?.segmentation?.backend === 'dual'
      if (isDual) {
        segChart = new uPlot({
          width: chartWidth('analytics-seg-chart'), height: 120,
          series: [
            {},
            { label: 'Dual', stroke: '#00ff88', fill: '#00ff8840', width: 1 },
            { label: 'FastSAM', stroke: '#1e90ff', fill: '#1e90ff40', width: 1 },
            { label: 'YOLOE', stroke: '#ffaa00', fill: '#ffaa0040', width: 1 },
          ],
          axes: [DARK_AXES, DARK_AXES_Y],
          cursor: { show: true },
          legend: { show: false },
        }, [[], [], [], []], el)
      } else {
        segChart = new uPlot({
          width: chartWidth('analytics-seg-chart'), height: 120,
          series: [
            {},
            { label: 'Masks', stroke: '#1e90ff', fill: '#1e90ff40', width: 2 },
          ],
          axes: [DARK_AXES, DARK_AXES_Y],
          cursor: { show: true },
          legend: { show: false },
        }, [[], []], el)
      }
    }
  }
}

// Auto-resize charts when viewport changes
const resizeObserver = new ResizeObserver(() => {
  if (!analyticsVisible) return
  if (throughputChart) throughputChart.setSize({ width: chartWidth('analytics-throughput-chart'), height: 150 })
  if (latencyChart) latencyChart.setSize({ width: chartWidth('analytics-latency-chart'), height: 150 })
  if (segChart) segChart.setSize({ width: chartWidth('analytics-seg-chart'), height: 120 })
})
const analyticsContent = document.getElementById('analytics-content')
if (analyticsContent) resizeObserver.observe(analyticsContent)

// ---- Chart data stacking helper ----

function stackSeries(arrays: number[][]): number[][] {
  if (arrays.length === 0) return []
  const stacked: number[][] = [arrays[0].slice()]
  for (let i = 1; i < arrays.length; i++) {
    stacked.push(stacked[i - 1].map((v, j) => v + (arrays[i][j] ?? 0)))
  }
  return stacked
}

function updateCharts() {
  // Throughput chart (lines, no stacking)
  if (throughputChart && latencyHistory.length > 0) {
    throughputChart.setData([
      latencyHistory.map(b => b.wall_ts),
      latencyHistory.map(b => b.input_hz ?? 0),
      latencyHistory.map(b => b.processing_hz ?? 0),
      latencyHistory.map(b => b.queue_depth_max ?? 0),
    ])
  }

  // Latency chart (stacked area)
  if (latencyChart && latencyHistory.length > 0) {
    const ts = latencyHistory.map(b => b.wall_ts)
    const raw = [
      latencyHistory.map(b => b.t_seg_ms ?? 0),
      latencyHistory.map(b => b.t_heur_ms ?? 0),
      latencyHistory.map(b => b.t_scoring_ms ?? 0),
      latencyHistory.map(b => b.t_clip_ms ?? 0),
      latencyHistory.map(b => b.t_assoc_ms ?? 0),
    ]
    const stacked = stackSeries(raw)
    latencyChart.setData([ts, ...stacked])
  }

  // Segmentation chart
  if (segChart && segHistory.length > 0) {
    const ts = segHistory.map(b => b.wall_ts)
    const isDual = runtimeConfig?.segmentation?.backend === 'dual'
    if (isDual) {
      const raw = [
        segHistory.map(b => b.dual_rate ?? 0),
        segHistory.map(b => b.fastsam_only_rate ?? 0),
        segHistory.map(b => b.yoloe_only_rate ?? 0),
      ]
      const stacked = stackSeries(raw)
      segChart.setData([ts, ...stacked])
    } else {
      segChart.setData([ts, segHistory.map(b => b.mean_total ?? 0)])
    }
  }
}

// ---- Copy button ----

function buildAnalyticsSummary(): string {
  const now = new Date().toISOString().replace('T', ' ').substring(0, 19)
  const la = latestAggregate.latency
  const sa = latestAggregate.segmentation
  const cfg = runtimeConfig

  const backend = cfg?.segmentation?.backend ?? 'unknown'
  const device = cfg?.pipeline?.device ?? '?'

  const fmtT = (s: any) => s ? `${(s.mean * 1000).toFixed(0)}ms / ${(s.p95 * 1000).toFixed(0)}ms` : '?'

  const lines = [
    `RTSM Analytics Snapshot (${now})`,
    `Backend: ${backend} | Device: ${device}`,
    '---',
  ]

  if (la) {
    const lastB = latencyHistory.length > 0 ? latencyHistory[latencyHistory.length - 1] : null
    lines.push(
      `Throughput: ${la.input_hz ?? 0} Hz in -> ${la.processing_hz ?? 0} Hz proc (${((la.effective_ratio ?? 0) * 100).toFixed(0)}%)`,
      `Queue: ${lastB?.queue_depth_mean ?? 0} avg / ${lastB?.queue_depth_max ?? 0} max / 512 cap | Drops: ${lastB?.queue_drops ?? 0}`,
      '---',
      'Latency (mean / p95):',
      `  Total: ${fmtT(la.t_total)}`,
      `  Seg: ${fmtT(la.t_segmentation)} | Heur: ${fmtT(la.t_heuristics)}`,
      `  Score: ${fmtT(la.t_scoring)} | CLIP: ${fmtT(la.t_clip)} | Assoc: ${fmtT(la.t_association)}`,
    )
  }

  if (sa) {
    lines.push('---')
    if (backend === 'dual') {
      lines.push(
        `Segmentation: Dual ${((sa.dual_rate ?? 0) * 100).toFixed(0)}% | FastSAM-only ${((sa.fastsam_only_rate ?? 0) * 100).toFixed(0)}% | YOLOE-only ${((sa.yoloe_only_rate ?? 0) * 100).toFixed(0)}%`,
        `Raw: FastSAM ${sa.mean_fastsam_raw ?? 0} avg | YOLOE ${sa.mean_yoloe_raw ?? 0} avg | Survival: ${((sa.staged_survival_rate ?? 0) * 100).toFixed(0)}%`,
      )
    } else {
      lines.push(
        `Segmentation (${backend}): ${sa.mean_total ?? 0} masks/frame | Survival: ${((sa.staged_survival_rate ?? 0) * 100).toFixed(0)}%`,
      )
    }
  }

  // Objects / association — session totals from Tier 2 history
  if (latencyHistory.length > 0) {
    // Find latest WM snapshot
    let wm = { confirmed: 0, total: 0, proto: 0 }
    for (let i = latencyHistory.length - 1; i >= 0; i--) {
      if ((latencyHistory[i].wm_total ?? 0) > 0) {
        wm = { confirmed: latencyHistory[i].wm_confirmed ?? 0, total: latencyHistory[i].wm_total ?? 0, proto: latencyHistory[i].wm_proto ?? 0 }
        break
      }
    }
    let totalMatched = 0, totalCreated = 0
    for (const b of latencyHistory) {
      totalMatched += b.assoc_matched ?? 0
      totalCreated += b.assoc_created ?? 0
    }
    lines.push(
      '---',
      `Objects: ${wm.confirmed} confirmed / ${wm.total} total (${wm.proto} proto)`,
      `Association: ${totalMatched} matched / ${totalCreated} created`,
    )
  }

  return lines.join('\n')
}

document.getElementById('analytics-copy')?.addEventListener('click', async () => {
  const btn = document.getElementById('analytics-copy')
  const text = buildAnalyticsSummary()
  try {
    await navigator.clipboard.writeText(text)
    if (btn) {
      btn.textContent = 'Copied!'
      btn.classList.add('copied')
      setTimeout(() => { btn.textContent = 'Copy Data'; btn.classList.remove('copied') }, 1500)
    }
  } catch {
    // Fallback for non-HTTPS
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
    if (btn) {
      btn.textContent = 'Copied!'
      btn.classList.add('copied')
      setTimeout(() => { btn.textContent = 'Copy Data'; btn.classList.remove('copied') }, 1500)
    }
  }
})

// Initialize
connectWebSocket()
updateHud()
updateConnectionButtons()  // Set initial button states
applyFlipScale()  // Apply default X/Y flip
animate()
