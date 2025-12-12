declare module 'three';
declare module 'three/examples/jsm/controls/OrbitControls.js' {
  import { Camera } from 'three'
  import { EventDispatcher } from 'three'
  export class OrbitControls extends EventDispatcher {
    constructor(object: Camera, domElement?: HTMLElement)
    enabled: boolean
    target: import('three').Vector3
    minPolarAngle: number
    maxPolarAngle: number
    minAzimuthAngle: number
    maxAzimuthAngle: number
    update(): void
    getPolarAngle(): number
    getAzimuthalAngle(): number
    addEventListener(type: string, listener: (event?: any) => void): void
    removeEventListener(type: string, listener: (event?: any) => void): void
  }
}
declare module 'three/examples/jsm/loaders/PLYLoader.js' {
  import { BufferGeometry, LoadingManager } from 'three'
  export class PLYLoader {
    constructor(manager?: LoadingManager)
    load(url: string, onLoad: (geometry: BufferGeometry) => void, onProgress?: (e: ProgressEvent) => void, onError?: (e?: any) => void): void
  }
}

