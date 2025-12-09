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

const meshes = new Map<string, THREE.Points>()
let wsConnected = false
let meshCreateCount = 0
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
      // Binary message - mesh_create
      const parsed = parseMeshCreate(ev.data)
      if (parsed) {
        createOrUpdateMesh(parsed.meshId, parsed.positions, parsed.colors, parsed.pose)
        updateHud()
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

    // Label - use object ID as primary identifier
    let label = objectLabels.get(obj.id)
    const labelText = obj.id.slice(0, 8)
    if (!label) {
      label = createTextSprite(labelText, obj.confirmed ? '#00ff88' : '#ffaa00')
      world.add(label)
      objectLabels.set(obj.id, label)
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

  // Create selection marker if needed
  if (!selectionMarker) {
    const geom = new THREE.RingGeometry(0.06, 0.08, 32)
    const mat = new THREE.MeshBasicMaterial({
      color: 0x1e90ff,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthTest: false,  // Always render on top
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

  // Update label text and position - use object ID as primary
  const labelText = selectedObj.id.slice(0, 8)
  // Recreate label sprite with new text
  const newLabel = createTextSprite(`▶ ${labelText}`, '#1e90ff')
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

// Gallery state
let gallerySnapshots: Array<{ index: number; data: string; size_bytes: number }> = []
let selectedSnapshotIndex = 0

function setMode(text: string) {
  if (modeEl) modeEl.textContent = `Mode: ${text}`
}
setMode('Free')

const initialCamPos = camera.position.clone()
const initialTarget = controls.target.clone()

let flippedX = true, flippedY = true, flippedZ = false

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

// Perform semantic search via CLIP + FAISS
async function performSemanticSearch() {
  const query = objectSearch?.value.trim()
  if (!query) {
    console.log('[semantic-search] Empty query, showing all objects')
    showAllObjects()
    return
  }

  searchQuery = query
  console.log(`[semantic-search] Searching for: "${query}"`)

  try {
    const resp = await fetch(`${RTSM_API_BASE}/search/semantic?query=${encodeURIComponent(query)}&top_k=5&threshold=0.10`)
    const data = await resp.json()
    console.log('[semantic-search] API response:', data)

    semanticSearchResults.clear()
    if (data.results) {
      for (const r of data.results) {
        semanticSearchResults.set(r.id, r.score)
      }
    }
    isSearchMode = true
    updateObjectList()
    updateObjectMarkers()  // Update 3D view to show only matches
    console.log(`[semantic-search] Found ${semanticSearchResults.size} matches`)
  } catch (e) {
    console.error('[semantic-search] API call failed:', e)
    alert('Semantic search failed. Is RTSM running?')
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
    const displayLabel = getBestDisplayLabel(obj)
    const isSelected = obj.id === selectedObjectId
    const semanticScore = semanticSearchResults.get(obj.id)

    // Show semantic similarity score if in search mode, otherwise show stability
    const scoreStr = isSearchMode && semanticScore !== undefined
      ? `sim: ${semanticScore.toFixed(2)}`
      : `stab: ${obj.stability.toFixed(2)}`

    return `
      <div class="object-item ${isSelected ? 'selected' : ''}" data-id="${obj.id}">
        <div class="object-dot ${obj.confirmed ? 'confirmed' : 'proto'}"></div>
        <div class="object-info">
          <div class="object-label">${obj.id.slice(0, 8)}</div>
          <div class="object-meta">${displayLabel} · ${scoreStr}</div>
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
    const obj = rtsmObjects.find(o => o.id === objectId)
    const label = obj ? getBestDisplayLabel(obj) : objectId.slice(0, 8)
    galleryTitle.textContent = `Snapshots: ${label}`
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

// Helper to update disconnect/reconnect button states
function updateConnectionButtons() {
  if (disconnectBtn) {
    disconnectBtn.disabled = !wsConnected
  }
  if (reconnectBtn) {
    reconnectBtn.disabled = wsConnected
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

const RTSM_API_BASE = '/api'  // Vite proxies /api to RTSM API server (port 8000)

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
  renderer.render(scene, camera)
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})

// Initialize
connectWebSocket()
updateHud()
updateConnectionButtons()  // Set initial button states
applyFlipScale()  // Apply default X/Y flip
animate()
