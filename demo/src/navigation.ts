/**
 * Viewport navigation on top of three.js OrbitControls: mouse schemes (three.js default, Blender, Maya),
 * zoom to the cursor, inertia, orbit pivot under the cursor, numpad view presets, frame-all / frame-selected
 * with a short tween, keyboard panning, and a key-binding help panel.
 *
 * The scheme only changes how the mouse buttons and modifiers map onto OrbitControls' rotate / pan / dolly;
 * the camera model stays a turntable (up axis locked), which is also Blender's default. Orthographic view
 * (Blender numpad 5) is not provided: the rest of the dashboard addresses one PerspectiveCamera.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export type Scheme = 'default' | 'blender' | 'maya'

export interface NavigationDeps {
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  dom: HTMLElement
  pointSources: () => THREE.Points[]
  markers: () => THREE.Object3D[]
  selectedPoint: () => THREE.Vector3 | null
  helpEl?: HTMLElement | null
  schemeSelect?: HTMLSelectElement | null
}

export interface Navigation {
  readonly scheme: Scheme
  setScheme(s: Scheme): void
  orbitButton(): number
  frameAll(): boolean
  frameSelected(): boolean
  preset(name: PresetName): void
  resetView(position: THREE.Vector3, target: THREE.Vector3): void
  update(): void
  toggleHelp(force?: boolean): void
  helpHtml(): string
}

export type PresetName = 'front' | 'back' | 'right' | 'left' | 'top' | 'bottom' | 'opposite'

const STORAGE_KEY = 'rtsm.nav.scheme'
const TWEEN_MS = 280
const ORBIT_STEP = THREE.MathUtils.degToRad(15)

const SCHEME_LABEL: Record<Scheme, string> = { default: 'three.js', blender: 'Blender', maya: 'Maya' }

function loadScheme(): Scheme {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'default' || v === 'blender' || v === 'maya') return v
  } catch { /* storage unavailable */ }
  return 'default'
}

function saveScheme(s: Scheme) {
  try { localStorage.setItem(STORAGE_KEY, s) } catch { /* storage unavailable */ }
}

function isTypingTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null
  if (!el || !el.tagName) return false
  const tag = el.tagName.toLowerCase()
  return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable
}

function easeOutCubic(x: number): number { return 1 - Math.pow(1 - x, 3) }

export function installNavigation(deps: NavigationDeps): Navigation {
  const { camera, controls, dom } = deps
  const MOUSE = THREE.MOUSE
  const buttons = controls.mouseButtons as unknown as Record<'LEFT' | 'MIDDLE' | 'RIGHT', THREE.MOUSE | null | undefined>

  // ---- shared behaviour, every scheme -------------------------------------------------------------
  controls.enableDamping = true
  controls.dampingFactor = 0.12
  ;(controls as unknown as { zoomToCursor: boolean }).zoomToCursor = true
  controls.screenSpacePanning = true
  controls.keyPanSpeed = 12
  controls.keys = { LEFT: 'ArrowLeft', UP: 'ArrowUp', RIGHT: 'ArrowRight', BOTTOM: 'ArrowDown' }
  controls.listenToKeyEvents(window as unknown as HTMLElement)

  let scheme: Scheme = loadScheme()

  function applyBaseMapping() {
    if (scheme === 'default') {
      buttons.LEFT = MOUSE.ROTATE; buttons.MIDDLE = MOUSE.DOLLY; buttons.RIGHT = MOUSE.PAN
    } else if (scheme === 'blender') {
      buttons.LEFT = null; buttons.MIDDLE = MOUSE.ROTATE; buttons.RIGHT = null
    } else {
      buttons.LEFT = null; buttons.MIDDLE = null; buttons.RIGHT = null
    }
  }

  function orbitButton(): number { return scheme === 'blender' ? 1 : 0 }

  // ---- pivot under the cursor (Blender "auto depth"): orbit about the surface depth under the pointer, without
  //      moving the view. The pivot slides along the view axis to the hit depth; the camera does not turn.
  const raycaster = new THREE.Raycaster()
  const ndc = new THREE.Vector2()
  const tmpDir = new THREE.Vector3()
  const tmpHit = new THREE.Vector3()

  function hitUnderPointer(clientX: number, clientY: number): THREE.Vector3 | null {
    const rect = dom.getBoundingClientRect()
    ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1
    ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1
    raycaster.setFromCamera(ndc, camera)
    raycaster.params.Points = { threshold: 0.012 }
    let best: THREE.Intersection | null = null
    const consider = (hits: THREE.Intersection[]) => { for (const h of hits) if (!best || h.distance < best.distance) best = h }
    consider(raycaster.intersectObjects(deps.markers(), false))
    for (const p of deps.pointSources()) consider(raycaster.intersectObject(p, false))
    return best ? tmpHit.copy((best as THREE.Intersection).point) : null
  }

  function pivotToDepthUnderPointer(clientX: number, clientY: number) {
    const hit = hitUnderPointer(clientX, clientY)
    if (!hit) return
    const depth = hit.distanceTo(camera.position)
    tmpDir.subVectors(controls.target, camera.position).normalize()
    controls.target.copy(camera.position).addScaledVector(tmpDir, depth)
  }

  // ---- per-press mapping from modifiers ---------------------------------------------------------
  dom.addEventListener('pointerdown', (e: PointerEvent) => {
    applyBaseMapping()
    if (scheme === 'blender') {
      if (e.button === 1) buttons.MIDDLE = e.shiftKey ? MOUSE.PAN : e.ctrlKey ? MOUSE.DOLLY : MOUSE.ROTATE
      if (e.button === 0 && e.altKey) buttons.LEFT = e.shiftKey ? MOUSE.PAN : e.ctrlKey ? MOUSE.DOLLY : MOUSE.ROTATE   // emulated 3-button mouse
    } else if (scheme === 'maya') {
      if (e.altKey) { buttons.LEFT = MOUSE.ROTATE; buttons.MIDDLE = MOUSE.PAN; buttons.RIGHT = MOUSE.DOLLY }
    }
    const mapped = e.button === 0 ? buttons.LEFT : e.button === 1 ? buttons.MIDDLE : buttons.RIGHT
    if (mapped === MOUSE.ROTATE || mapped === MOUSE.DOLLY) pivotToDepthUnderPointer(e.clientX, e.clientY)
    if (e.button === 1) e.preventDefault()   // no autoscroll cursor on middle press
  }, { capture: true })

  dom.addEventListener('pointerup', () => { applyBaseMapping() }, { capture: true })

  // Blender: Shift+wheel pans vertically, Ctrl+wheel pans horizontally; the plain wheel zooms to the cursor.
  dom.addEventListener('wheel', (e: WheelEvent) => {
    if (scheme !== 'blender' || !(e.shiftKey || e.ctrlKey)) return
    e.preventDefault()
    e.stopImmediatePropagation()
    const dist = camera.position.distanceTo(controls.target)
    const step = -Math.sign(e.deltaY) * dist * 0.06
    // the camera's own up (Shift) or right (Ctrl) axis, in world space
    const axis = (e.shiftKey ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)).applyQuaternion(camera.quaternion).normalize()
    camera.position.addScaledVector(axis, step)
    controls.target.addScaledVector(axis, step)
    controls.update()
  }, { capture: true, passive: false })

  // ---- tweened moves (presets, framing) ---------------------------------------------------------
  let tween: { p0: THREE.Vector3; p1: THREE.Vector3; t0: THREE.Vector3; t1: THREE.Vector3; start: number } | null = null

  function moveTo(position: THREE.Vector3, target: THREE.Vector3) {
    tween = { p0: camera.position.clone(), p1: position.clone(), t0: controls.target.clone(), t1: target.clone(), start: performance.now() }
  }

  function update() {
    if (tween) {
      const k = Math.min(1, (performance.now() - tween.start) / TWEEN_MS)
      const e = easeOutCubic(k)
      camera.position.lerpVectors(tween.p0, tween.p1, e)
      controls.target.lerpVectors(tween.t0, tween.t1, e)
      if (k >= 1) tween = null
    }
    controls.update()
  }

  // ---- framing -------------------------------------------------------------------------------------
  const box = new THREE.Box3()
  const tmpBox = new THREE.Box3()
  const center = new THREE.Vector3()
  const size = new THREE.Vector3()

  function fitDistance(radius: number): number {
    const vFov = THREE.MathUtils.degToRad(camera.fov)
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect)
    const fov = Math.min(vFov, hFov)
    return Math.max(0.3, (radius / Math.sin(fov / 2)) * 1.08)
  }

  function frameSphere(c: THREE.Vector3, radius: number) {
    tmpDir.subVectors(camera.position, controls.target)
    if (tmpDir.lengthSq() < 1e-9) tmpDir.set(1, 1, 1)
    tmpDir.normalize()
    const dist = fitDistance(radius)
    moveTo(c.clone().addScaledVector(tmpDir, dist), c)
    camera.far = Math.max(camera.far, dist * 10)
    camera.updateProjectionMatrix()
  }

  function frameAll(): boolean {
    box.makeEmpty()
    for (const p of deps.pointSources()) {
      const g = p.geometry
      if (!g.boundingBox) g.computeBoundingBox()
      if (!g.boundingBox) continue
      tmpBox.copy(g.boundingBox).applyMatrix4(p.matrixWorld)
      box.union(tmpBox)
    }
    for (const m of deps.markers()) box.expandByPoint(m.getWorldPosition(tmpHit))
    if (box.isEmpty()) return false
    box.getCenter(center)
    box.getSize(size)
    frameSphere(center, Math.max(0.25, size.length() / 2))
    return true
  }

  function frameSelected(): boolean {
    const p = deps.selectedPoint()
    if (!p) return false
    frameSphere(p, 0.6)
    return true
  }

  // ---- view presets, in the camera's up basis (Y up by default; the world group carries any axis flip) --
  function basis() {
    const up = camera.up.clone().normalize()
    const front = Math.abs(up.y) > 0.5 ? new THREE.Vector3(0, 0, 1) : Math.abs(up.z) > 0.5 ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(0, 0, 1)
    const right = new THREE.Vector3().crossVectors(up, front).normalize()
    front.crossVectors(right, up).normalize()
    return { up, front, right }
  }

  function preset(name: PresetName) {
    const { up, front, right } = basis()
    const dist = Math.max(0.3, camera.position.distanceTo(controls.target))
    const dir = new THREE.Vector3()
    switch (name) {
      case 'front': dir.copy(front); break
      case 'back': dir.copy(front).negate(); break
      case 'right': dir.copy(right); break
      case 'left': dir.copy(right).negate(); break
      case 'top': dir.copy(up).addScaledVector(front, 1e-3); break
      case 'bottom': dir.copy(up).negate().addScaledVector(front, 1e-3); break
      case 'opposite': dir.subVectors(camera.position, controls.target).negate(); break
    }
    dir.normalize()
    moveTo(controls.target.clone().addScaledVector(dir, dist), controls.target)
  }

  function orbitStep(yaw: number, pitch: number) {
    const offset = new THREE.Vector3().subVectors(camera.position, controls.target)
    const up = camera.up.clone().normalize()
    if (yaw) offset.applyAxisAngle(up, yaw)
    if (pitch) {
      const right = new THREE.Vector3().crossVectors(up, offset).normalize()
      const polar = offset.angleTo(up)
      const next = THREE.MathUtils.clamp(polar + pitch, 0.02, Math.PI - 0.02)
      offset.applyAxisAngle(right, polar - next)
    }
    moveTo(controls.target.clone().add(offset), controls.target)
  }

  function resetView(position: THREE.Vector3, target: THREE.Vector3) {
    tween = null
    camera.position.copy(position)
    controls.target.copy(target)
    controls.update()
  }

  // ---- keyboard --------------------------------------------------------------------------------------
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (isTypingTarget(e.target) || e.metaKey) return
    const code = e.code
    const digit = (code.startsWith('Numpad') ? code.slice(6) : code.startsWith('Digit') ? code.slice(5) : '')
    let handled = true
    if (digit === '1') preset(e.ctrlKey ? 'back' : 'front')
    else if (digit === '3') preset(e.ctrlKey ? 'left' : 'right')
    else if (digit === '7') preset(e.ctrlKey ? 'bottom' : 'top')
    else if (digit === '9') preset('opposite')
    else if (digit === '4') orbitStep(ORBIT_STEP, 0)
    else if (digit === '6') orbitStep(-ORBIT_STEP, 0)
    else if (digit === '8') orbitStep(0, -ORBIT_STEP)
    else if (digit === '2') orbitStep(0, ORBIT_STEP)
    else if (code === 'NumpadDecimal' || code === 'Period' || code === 'KeyF') { if (!frameSelected()) frameAll() }
    else if (code === 'Home' || code === 'KeyA' && e.shiftKey) frameAll()
    else if (code === 'Slash' && e.shiftKey || code === 'KeyH') toggleHelp()
    else handled = false
    if (handled) e.preventDefault()
  })

  // ---- help panel ------------------------------------------------------------------------------------
  function helpHtml(): string {
    const mouse: Record<Scheme, string[]> = {
      default: ['Left drag: orbit', 'Right drag: pan', 'Middle drag or wheel: zoom (to the cursor)', 'Shift + left drag: pan'],
      blender: ['Middle drag: orbit', 'Shift + middle drag: pan', 'Ctrl + middle drag: zoom', 'Wheel: zoom to the cursor', 'Shift + wheel: pan up / down', 'Ctrl + wheel: pan left / right', 'Alt + left drag: orbit (emulated 3-button mouse)'],
      maya: ['Alt + left drag: tumble', 'Alt + middle drag: track', 'Alt + right drag: dolly', 'Wheel: zoom to the cursor'],
    }
    const keys = ['Numpad 1 / 3 / 7: front / right / top (Ctrl: back / left / bottom)', 'Numpad 9: opposite side', 'Numpad 2 / 4 / 6 / 8: orbit 15°',
      'F or Numpad .: frame the selected object (else everything)', 'Home or Shift+A: frame everything', 'Arrows: pan',
      'Double-click: set the pivot on a point', 'Z / X / C while dragging: pitch-only / yaw-only / roll', 'H or ?: this panel']
    const li = (s: string) => `<li>${s}</li>`
    return `<div class="nav-help-title">Navigation · ${SCHEME_LABEL[scheme]}</div><div class="nav-help-cols"><div><div class="nav-help-sub">Mouse</div><ul>${mouse[scheme].map(li).join('')}</ul></div>`
      + `<div><div class="nav-help-sub">Keys</div><ul>${keys.map(li).join('')}</ul></div></div>`
      + `<div class="nav-help-note">Orbit pivots at the depth under the cursor; the view does not jump. The up axis stays locked (turntable); use Flip X / Y / Z for the world's up.</div>`
  }

  function toggleHelp(force?: boolean) {
    const el = deps.helpEl
    if (!el) return
    const show = force !== undefined ? force : el.style.display === 'none' || !el.style.display
    el.innerHTML = helpHtml()
    el.style.display = show ? 'block' : 'none'
  }

  function setScheme(s: Scheme) {
    scheme = s
    saveScheme(s)
    applyBaseMapping()
    if (deps.schemeSelect && deps.schemeSelect.value !== s) deps.schemeSelect.value = s
    if (deps.helpEl && deps.helpEl.style.display === 'block') deps.helpEl.innerHTML = helpHtml()
  }

  if (deps.schemeSelect) {
    deps.schemeSelect.value = scheme
    deps.schemeSelect.addEventListener('change', () => setScheme(deps.schemeSelect!.value as Scheme))
  }
  if (deps.helpEl) deps.helpEl.style.display = 'none'
  applyBaseMapping()

  return {
    get scheme() { return scheme },
    setScheme, orbitButton, frameAll, frameSelected, preset, resetView, update, toggleHelp, helpHtml,
  }
}
