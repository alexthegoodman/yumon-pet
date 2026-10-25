// Yumon Universe: Three.js Simulation World for Yumon Portal
(function () {
  'use strict';

  // Constants for Suburban mode
  const EXTENT = 30.0;
  const SIGHT = 14.0;
  const MAX_BUILD_TIER = 60;

  // Aquarium Tank Dimensions (Rectangular desktop tank)
  const TANK_WIDTH = 56.0;   // X: -28.0 to 28.0
  const TANK_DEPTH = 20.0;   // Z: -10.0 to 10.0
  const TANK_HEIGHT = 20.5;  // Y: 0 to 20.5
  const WATER_LEVEL = 19.4;  // Top surface of water

  // Recursive WebGL disposal helper to prevent resource leaks
  function disposeHierarchy(obj) {
    if (!obj) return;
    const disposedGeos = new Set();
    const disposedMats = new Set();

    obj.traverse((child) => {
      if (child.geometry && !child.geometry.userData?.isShared) {
        if (!disposedGeos.has(child.geometry)) {
          disposedGeos.add(child.geometry);
          child.geometry.dispose();
        }
      }
      if (child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (let i = 0; i < mats.length; i++) {
          const m = mats[i];
          if (m && !m.userData?.isShared) {
            if (!disposedMats.has(m)) {
              disposedMats.add(m);
              m.dispose();
            }
          }
        }
      }
    });
  }
  function createRandom(seed) {
    let state = BigInt(seed);
    return function () {
      state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
      let n = state;
      n = ((n ^ (n >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
      n = ((n ^ (n >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
      return Number((n ^ (n >> 31n)) >> 40n) / (1 << 24);
    };
  }

  const Kinds = {
    House: 0,
    Shop: 1,
    Tree: 2,
    Flowers: 3,
    Bench: 4,
    Ball: 5,
    Garden: 6,
    Mailbox: 7,
    Pond: 8,
    Business: 9, // Suburban Tower
    Church: 10,  // Suburban Temple
    Telescope: 11, // Suburban Observatory
  };

  const Behaviors = {
    Rest: 'Rest',
    Visit: 'Visit',
    Play: 'Play',
    Collect: 'Collect',
    Tend: 'Tend',
    Inspect: 'Inspect',
    Explore: 'Explore',
    Build: 'Build',
    Eat: 'Eat',
  };

  const BehaviorLabels = {
    Suburban: {
      Rest: 'resting ☕',
      Visit: 'visiting 🚪',
      Play: 'playing ⚽',
      Collect: 'collecting 🌸',
      Tend: 'gardening 🌱',
      Inspect: 'looking around 🔍',
      Explore: 'exploring 🧭',
      Build: 'building 🔨',
      Eat: 'having a snack 🥪',
    },
    FishTank: {
      Rest: 'resting in rock cave 🪨',
      Visit: 'cruising by driftwood 🪵',
      Play: 'riding water current 🌊',
      Collect: 'grazing on algae 🌿',
      Tend: 'nibbling moss rock 🌱',
      Inspect: 'peering through glass 👀',
      Explore: 'swimming the open water 🐠',
      Build: 'playing near air stone 🫧',
      Eat: 'chasing food flakes 🦐',
    },
  };

  const Nouns = {
    Suburban: {
      [Kinds.House]: 'house',
      [Kinds.Shop]: 'shop',
      [Kinds.Tree]: 'tree',
      [Kinds.Flowers]: 'flowers',
      [Kinds.Bench]: 'bench',
      [Kinds.Ball]: 'ball',
      [Kinds.Garden]: 'garden',
      [Kinds.Mailbox]: 'mailbox',
      [Kinds.Pond]: 'pond',
      [Kinds.Business]: 'tower',
      [Kinds.Church]: 'temple',
      [Kinds.Telescope]: 'telescope',
    },
    FishTank: {
      0: 'slate cave',
      1: 'dragon stone',
      2: 'spiderwood root',
      3: 'driftwood branch',
      4: 'vallisneria forest',
      5: 'anubias plant',
      6: 'air stone bubbler',
      7: 'river pebble bed',
      8: 'mossy boulder',
      9: 'surface current',
      10: 'top corner current',
      11: 'aquarium bed',
    },
  };

  function getSuburbanRadius(kind) {
    switch (kind) {
      case Kinds.House:
      case Kinds.Shop:
      case Kinds.Business:
        return 2.4;
      case Kinds.Church:
        return 2.6;
      case Kinds.Pond:
        return 1.6;
      case Kinds.Telescope:
        return 1.8;
      case Kinds.Tree:
        return 0.5;
      default:
        return 0.4;
    }
  }

  function getVerbAndBehavior(kind, theme) {
    if (theme === 'FishTank') {
      switch (kind % 6) {
        case 0:
          return { verb: 'rest in', behavior: Behaviors.Rest };
        case 1:
          return { verb: 'inspect', behavior: Behaviors.Inspect };
        case 2:
          return { verb: 'graze on', behavior: Behaviors.Collect };
        case 3:
          return { verb: 'ride current by', behavior: Behaviors.Play };
        case 4:
          return { verb: 'swim near', behavior: Behaviors.Visit };
        default:
          return { verb: 'explore', behavior: Behaviors.Explore };
      }
    }
    switch (kind) {
      case Kinds.Bench:
        return { verb: 'rest by', behavior: Behaviors.Rest };
      case Kinds.Ball:
        return { verb: 'play with', behavior: Behaviors.Play };
      case Kinds.Garden:
        return { verb: 'tend', behavior: Behaviors.Tend };
      case Kinds.Flowers:
        return { verb: 'collect', behavior: Behaviors.Collect };
      case Kinds.House:
      case Kinds.Shop:
        return { verb: 'visit', behavior: Behaviors.Visit };
      case Kinds.Business:
      case Kinds.Church:
      case Kinds.Telescope:
        return { verb: 'build', behavior: Behaviors.Build };
      default:
        return { verb: 'look at', behavior: Behaviors.Inspect };
    }
  }

  // Generate Suburban Neighborhood Grid
  function generateNeighborhood(seed) {
    const random = createRandom(seed);
    const places = [];

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const x = -22.5 + col * 15.0;
        const z = -22.5 + row * 15.0;

        let mainKind = Kinds.House;
        if (row === 1 && col === 1) mainKind = Kinds.Pond;
        else if (row === 0 && col === 3) mainKind = Kinds.Church;
        else if (row === 3 && col === 0) mainKind = Kinds.Telescope;
        else if (
          (row === 0 && col === 0) ||
          (row === 1 && col === 3) ||
          (row === 2 && col === 1) ||
          (row === 3 && col === 3)
        )
          mainKind = Kinds.Business;
        else if ((row === 2 && col === 3) || (row === 3 && col === 2))
          mainKind = Kinds.Shop;

        const tint = [
          Math.floor(140 + random() * 90),
          Math.floor(140 + random() * 80),
          Math.floor(120 + random() * 90),
        ];

        places.push({
          id: places.length,
          kind: mainKind,
          pos: { x: x + random() - 0.5, z: z + random() - 0.5 },
          radius: getSuburbanRadius(mainKind),
          tint,
          activityCount: 0,
        });

        const slots = [
          Kinds.Tree,
          Kinds.Flowers,
          Kinds.Bench,
          Kinds.Ball,
          Kinds.Garden,
          Kinds.Mailbox,
        ];

        for (let s = 0; s < slots.length; s++) {
          const angle = (s * Math.PI * 2.0) / 6.0;
          const radius = 4.3 + random() * 0.5;
          const slotKind = slots[s];
          places.push({
            id: places.length,
            kind: slotKind,
            pos: {
              x: x + Math.cos(angle) * radius,
              z: z + Math.sin(angle) * radius,
            },
            radius: getSuburbanRadius(slotKind),
            tint,
            activityCount: 0,
          });
        }
      }
    }

    return { seed, places };
  }

  // Generate 3D Natural Aquascape Features (No houses, just rocks, driftwood, plants)
  function generateAquascapeFeatures() {
    return [
      { id: 0, name: 'slate cave', pos: { x: 0, y: 3.2, z: -1.5 }, radius: 3.2 },
      { id: 1, name: 'dragon stone', pos: { x: 15, y: 6.0, z: -3.0 }, radius: 3.0 },
      { id: 2, name: 'spiderwood root', pos: { x: -8, y: 2.8, z: 2.2 }, radius: 2.5 },
      { id: 3, name: 'driftwood branch', pos: { x: 8, y: 3.2, z: 1.8 }, radius: 2.5 },
      { id: 4, name: 'vallisneria forest', pos: { x: 21, y: 9.0, z: -4.5 }, radius: 3.5 },
      { id: 5, name: 'anubias plant', pos: { x: -14, y: 5.2, z: -2.5 }, radius: 2.8 },
      { id: 6, name: 'air stone bubbler', pos: { x: -20, y: 1.2, z: -5.0 }, radius: 2.0 },
      { id: 7, name: 'river pebble bed', pos: { x: 0, y: 1.2, z: 3.8 }, radius: 2.8 },
      { id: 8, name: 'mossy boulder', pos: { x: -16, y: 2.5, z: 1.0 }, radius: 2.6 },
      { id: 9, name: 'surface current', pos: { x: 0, y: 17.5, z: 0 }, radius: 4.0 },
      { id: 10, name: 'top corner current', pos: { x: -18, y: 17.2, z: 1 }, radius: 3.5 },
      { id: 11, name: 'aquarium bed', pos: { x: 18, y: 1.4, z: 2.5 }, radius: 2.5 },
    ];
  }

  function isWalkableSuburban(pos, places) {
    if (Math.abs(pos.x) >= EXTENT - 0.6 || Math.abs(pos.z) >= EXTENT - 0.6) return false;
    for (let i = 0; i < places.length; i++) {
      const p = places[i];
      const dx = pos.x - p.pos.x;
      const dz = pos.z - p.pos.z;
      const dist = Math.hypot(dx, dz);
      if (dist <= p.radius + 0.35) return false;
    }
    return true;
  }

  function approachPlaceSuburban(from, place) {
    const dx = from.x - place.pos.x;
    const dz = from.z - place.pos.z;
    const dist = Math.max(0.01, Math.hypot(dx, dz));
    const offset = place.radius + 0.8;
    return {
      x: place.pos.x + (dx / dist) * offset,
      z: place.pos.z + (dz / dist) * offset,
    };
  }

  // Three.js Universe App
  class YumonUniverseApp {
    constructor(container) {
      this.container = container;
      this.seed = 42;
      this.theme = 'Suburban';
      this.world = generateNeighborhood(this.seed);
      this.aquascapePlaces = generateAquascapeFeatures();
      this.paused = false;
      this.isPainting = false;
      this.lastRenderTime = 0;
      this.selectedCitizen = 0;

      this.landmarks = [];
      this.sceneryGroup = null;
      this.territoryGroup = null;
      this.sparklesGroup = null;
      this.flakesGroup = null;
      this.ambientBubbles = [];
      this.swayingPlants = [];
      this.foodFlakes = [];
      this.flakesEatenTotal = 0;
      this.animId = null;

      this.citizens = [
        {
          name: 'Ember',
          kingdom: 0,
          color: 0xf27059,
          secondaryColor: 0xffd166,
          homeParcel: 0,
          role: 'Architect · Builder',
          fishRole: 'Flame Goby · Swimmer',
          savings: 42,
        },
        {
          name: 'Ripple',
          kingdom: 0,
          color: 0x4292f1,
          secondaryColor: 0x90e0ef,
          homeParcel: 1,
          role: 'Explorer · Scout',
          fishRole: 'Neon Tetra · Glider',
          savings: 38,
        },
        {
          name: 'Fern',
          kingdom: 1,
          color: 0x48b376,
          secondaryColor: 0xb7e4c7,
          homeParcel: 3,
          role: 'Botanist · Planter',
          fishRole: 'Emerald Cory · Grazer',
          savings: 55,
        },
        {
          name: 'Sol',
          kingdom: 2,
          color: 0xf5a623,
          secondaryColor: 0xffe3a8,
          homeParcel: 14,
          role: 'Astronomer · Guide',
          fishRole: 'Golden Guppy · Dasher',
          savings: 49,
        },
      ];

      this.yumons = [];
      this.initThree();
      this.buildWorld();
      this.initCitizens();
      this.initHUD();
      this.animate = this.animate.bind(this);
      requestAnimationFrame(this.animate);
    }

    initThree() {
      const width = this.container.clientWidth || 508;
      const height = this.container.clientHeight || 508;

      this.scene = new THREE.Scene();

      this.camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 300);

      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.container.appendChild(this.renderer.domElement);

      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.minDistance = 6;
      this.controls.maxDistance = 120;

      // Lights
      this.ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      this.scene.add(this.ambientLight);

      this.sunLight = new THREE.DirectionalLight(0xfff8ee, 1.2);
      this.sunLight.castShadow = true;
      this.sunLight.shadow.mapSize.width = 1024;
      this.sunLight.shadow.mapSize.height = 1024;
      this.sunLight.shadow.camera.near = 10;
      this.sunLight.shadow.camera.far = 120;
      this.sunLight.shadow.camera.left = -32;
      this.sunLight.shadow.camera.right = 32;
      this.sunLight.shadow.camera.top = 32;
      this.sunLight.shadow.camera.bottom = -32;
      this.sunLight.shadow.bias = -0.0005;
      this.scene.add(this.sunLight);

      this.hemiLight = new THREE.HemisphereLight(0xffffff, 0x4a6350, 0.35);
      this.scene.add(this.hemiLight);

      this.updateSkyAndFog();
      this.setCameraForTheme(this.theme);

      // Sparkles effect group & pool (fixed size to eliminate WebGL buffer churn)
      this.sparkleGeoAquarium = new THREE.SphereGeometry(0.11, 6, 6);
      this.sparkleGeoAquarium.userData = { isShared: true };
      this.sparkleGeoSuburban = new THREE.SphereGeometry(0.08, 6, 6);
      this.sparkleGeoSuburban.userData = { isShared: true };
      this.sparkleMaterialCache = new Map();

      this.sparklePoolSize = 128;
      this.sparklePoolIndex = 0;
      this.sparklePool = [];
      this.sparklesGroup = new THREE.Group();
      this.scene.add(this.sparklesGroup);

      const defaultSparkleMat = this.getSparkleMaterial(0xffd166, false);
      for (let i = 0; i < this.sparklePoolSize; i++) {
        const p = new THREE.Mesh(this.sparkleGeoSuburban, defaultSparkleMat);
        p.visible = false;
        p.userData = {
          active: false,
          vx: 0,
          vy: 0,
          vz: 0,
          life: 0,
        };
        this.sparklesGroup.add(p);
        this.sparklePool.push(p);
      }

      // Food Flakes pool (fixed size to eliminate WebGL buffer churn)
      this.flakeGeo = new THREE.CylinderGeometry(0.18, 0.22, 0.04, 6);
      this.flakeGeo.userData = { isShared: true };
      const flakeColors = [0xf59e0b, 0xef4444, 0x10b981, 0xf97316, 0xeab308, 0xec4899];
      this.flakeMaterials = flakeColors.map((color) => {
        const mat = new THREE.MeshStandardMaterial({
          color,
          roughness: 0.5,
          side: THREE.DoubleSide,
        });
        mat.userData = { isShared: true };
        return mat;
      });

      this.flakePoolSize = 64;
      this.flakePoolIndex = 0;
      this.foodFlakes = [];
      this.flakesGroup = new THREE.Group();
      this.scene.add(this.flakesGroup);

      for (let i = 0; i < this.flakePoolSize; i++) {
        const mat = this.flakeMaterials[i % this.flakeMaterials.length];
        const flakeMesh = new THREE.Mesh(this.flakeGeo, mat);
        flakeMesh.visible = false;
        flakeMesh.userData = {
          active: false,
          pos: { x: 0, y: 0, z: 0 },
          vy: 1.0,
          phase: 0,
          rotSpeed: 0,
          eaten: false,
          settledTime: 0,
        };
        this.flakesGroup.add(flakeMesh);
        this.foodFlakes.push(flakeMesh);
      }

      // Resize observer
      this.resizeObserver = new ResizeObserver(() => {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w && h) {
          this.camera.aspect = w / h;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(w, h);
        }
      });
      this.resizeObserver.observe(this.container);

      // Raycaster for selecting Yumons or dropping food
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();
      this.planeZ0 = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
      this.intersectPoint = new THREE.Vector3();
      this.renderer.domElement.addEventListener('pointerdown', (e) => {
        this.pointerDownTime = performance.now();
        this.pointerDownPos = { x: e.clientX, y: e.clientY };
      });
      this.renderer.domElement.addEventListener('pointerup', (e) => {
        const dt = performance.now() - (this.pointerDownTime || 0);
        const dist = Math.hypot(
          e.clientX - (this.pointerDownPos?.x || 0),
          e.clientY - (this.pointerDownPos?.y || 0)
        );
        if (dt < 250 && dist < 5) {
          this.onClick(e);
        }
      });
    }

    getSparkleMaterial(colorHex, isAquarium) {
      const color = colorHex || (isAquarium ? 0x7ce8ff : 0xffd166);
      const key = color * 2 + (isAquarium ? 1 : 0);
      let mat = this.sparkleMaterialCache.get(key);
      if (!mat) {
        mat = new THREE.MeshBasicMaterial({
          color,
          transparent: isAquarium,
          opacity: isAquarium ? 0.75 : 1.0,
        });
        mat.userData = { isShared: true };
        this.sparkleMaterialCache.set(key, mat);
      }
      return mat;
    }

    setCameraForTheme(theme) {
      if (theme === 'FishTank') {
        // Direct eye-level front/side view looking into the rectangular aquarium
        this.camera.position.set(0, 10.5, 38);
        this.controls.target.set(0, 9.5, 0);
        this.controls.maxPolarAngle = Math.PI / 2 + 0.08;
      } else {
        // Suburban: top-down isometric view
        this.camera.position.set(38, 42, 48);
        this.controls.target.set(0, 0, 0);
        this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
      }
      this.controls.update();
    }

    updateSkyAndFog() {
      if (this.theme === 'Suburban') {
        if (!this.scene.background) {
          this.scene.background = new THREE.Color(0xa3c9e0);
        } else {
          this.scene.background.setHex(0xa3c9e0);
        }
        if (!this.scene.fog) {
          this.scene.fog = new THREE.FogExp2(0xa3c9e0, 0.007);
        } else {
          this.scene.fog.color.setHex(0xa3c9e0);
          this.scene.fog.density = 0.007;
        }
        if (this.hemiLight) {
          this.hemiLight.color.setHex(0xffffff);
          this.hemiLight.groundColor.setHex(0x4a6350);
          this.hemiLight.intensity = 0.35;
        }
        if (this.sunLight) {
          this.sunLight.color.setHex(0xfff8ee);
          this.sunLight.intensity = 1.2;
          this.sunLight.position.set(28, 48, 22);
        }
      } else {
        // Real Fish Tank: Deep crystal-clear tropical aquarium water with top LED illumination
        if (!this.scene.background) {
          this.scene.background = new THREE.Color(0x0a263d);
        } else {
          this.scene.background.setHex(0x0a263d);
        }
        if (!this.scene.fog) {
          this.scene.fog = new THREE.FogExp2(0x0c314f, 0.01);
        } else {
          this.scene.fog.color.setHex(0x0c314f);
          this.scene.fog.density = 0.01;
        }
        if (this.hemiLight) {
          this.hemiLight.color.setHex(0x6ee7b7);
          this.hemiLight.groundColor.setHex(0x071e2e);
          this.hemiLight.intensity = 0.7;
        }
        if (this.sunLight) {
          // Top aquarium LED hood light directly above the rectangular water surface
          this.sunLight.color.setHex(0xe0f7ff);
          this.sunLight.intensity = 1.5;
          this.sunLight.position.set(0, 32, 0);
        }
      }
    }

    buildWorld() {
      if (this.sceneryGroup) {
        disposeHierarchy(this.sceneryGroup);
        this.scene.remove(this.sceneryGroup);
      }
      if (this.territoryGroup) {
        disposeHierarchy(this.territoryGroup);
        this.scene.remove(this.territoryGroup);
      }

      this.sceneryGroup = new THREE.Group();
      this.territoryGroup = new THREE.Group();
      this.landmarks = [];
      this.ambientBubbles = [];
      this.swayingPlants = [];

      // Deactivate any active food flakes
      if (this.foodFlakes) {
        for (let f = 0; f < this.foodFlakes.length; f++) {
          this.foodFlakes[f].userData.active = false;
          this.foodFlakes[f].visible = false;
        }
      }
      if (this.flakesGroup) {
        this.flakesGroup.visible = this.theme === 'FishTank';
      }

      if (this.theme === 'Suburban') {
        // ── Suburban Environment ──
        const groundMat = new THREE.MeshStandardMaterial({
          color: 0x73a35b,
          roughness: 0.9,
        });
        const groundGeo = new THREE.BoxGeometry(EXTENT * 2.0, 0.25, EXTENT * 2.0);
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.position.set(0, -0.125, 0);
        ground.receiveShadow = true;
        this.sceneryGroup.add(ground);

        // Roads & Sidewalks along axes [-15.0, 0.0, 15.0]
        const sidewalkMat = new THREE.MeshStandardMaterial({
          color: 0xc0beb2,
          roughness: 0.85,
        });
        const roadMat = new THREE.MeshStandardMaterial({
          color: 0x484f58,
          roughness: 0.75,
        });
        const dashMat = new THREE.MeshBasicMaterial({ color: 0xf0e3ad });

        const roadAxes = [-15.0, 0.0, 15.0];
        for (let i = 0; i < roadAxes.length; i++) {
          const axis = roadAxes[i];

          const swX = new THREE.Mesh(new THREE.BoxGeometry(5.0, 0.05, 60.0), sidewalkMat);
          swX.position.set(axis, 0.01, 0);
          swX.receiveShadow = true;
          this.sceneryGroup.add(swX);

          const swZ = new THREE.Mesh(new THREE.BoxGeometry(60.0, 0.05, 5.0), sidewalkMat);
          swZ.position.set(0, 0.01, axis);
          swZ.receiveShadow = true;
          this.sceneryGroup.add(swZ);

          const rdX = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.05, 60.0), roadMat);
          rdX.position.set(axis, 0.035, 0);
          rdX.receiveShadow = true;
          this.sceneryGroup.add(rdX);

          const rdZ = new THREE.Mesh(new THREE.BoxGeometry(60.0, 0.05, 3.2), roadMat);
          rdZ.position.set(0, 0.035, axis);
          rdZ.receiveShadow = true;
          this.sceneryGroup.add(rdZ);

          for (let n = -9; n <= 9; n++) {
            const v = n * 3.0;
            const dX = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.015, 1.0), dashMat);
            dX.position.set(axis, 0.07, v);
            this.sceneryGroup.add(dX);

            const dZ = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.015, 0.08), dashMat);
            dZ.position.set(v, 0.07, axis);
            this.sceneryGroup.add(dZ);
          }
        }

        // Suburban Territory Borders
        const kingdomColors = [0x4b9beb, 0xeba541, 0xa064d7];
        for (let row = 0; row < 4; row++) {
          for (let col = 0; col < 4; col++) {
            const pId = row * 4 + col;
            const px = -22.5 + col * 15.0;
            const pz = -22.5 + row * 15.0;
            let kOwner = null;
            if (pId === 0 || pId === 1) kOwner = 0;
            else if (pId === 3 || pId === 7) kOwner = 1;
            else if (pId === 14 || pId === 15) kOwner = 2;

            const bColor = kOwner !== null ? kingdomColors[kOwner] : 0x6e787d;
            const bMat = new THREE.MeshBasicMaterial({ color: bColor });

            for (const dz of [-7.4, 7.4]) {
              const bX = new THREE.Mesh(new THREE.BoxGeometry(14.8, 0.04, 0.1), bMat);
              bX.position.set(px, 0.06, pz + dz);
              this.territoryGroup.add(bX);
            }
            for (const dx of [-7.4, 7.4]) {
              const bZ = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.04, 14.8), bMat);
              bZ.position.set(px + dx, 0.06, pz);
              this.territoryGroup.add(bZ);
            }
          }
        }

        // Populate Suburban Scenery and Landmarks
        for (let i = 0; i < this.world.places.length; i++) {
          const place = this.world.places[i];
          if (
            place.kind === Kinds.Business ||
            place.kind === Kinds.Church ||
            place.kind === Kinds.Telescope
          ) {
            this.createSuburbanLandmark(place);
          } else {
            this.createSuburbanSceneryItem(place);
          }
        }
      } else {
        // ── Real Life Rectangular Fish Tank (No houses, just rocks, driftwood, plants) ──
        // Tank Dimensions: Width 56, Depth 20, Height 20.5 (Front/Side view)

        // 1. Natural Aquarium Gravel / Sand Bed (Sloped naturally)
        const sandMat = new THREE.MeshStandardMaterial({
          color: 0xe5d4ab,
          roughness: 0.95,
        });
        const sandBed = new THREE.Mesh(
          new THREE.BoxGeometry(TANK_WIDTH, 0.4, TANK_DEPTH),
          sandMat
        );
        sandBed.position.set(0, -0.2, 0);
        sandBed.receiveShadow = true;
        this.sceneryGroup.add(sandBed);

        // Scattered gravel mounds & smooth river pebbles along the bottom
        const pebbleMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 });
        const riverPebbleColors = [0x64748b, 0x94a3b8, 0x475569, 0x78716c, 0xa8a29e, 0x334155];
        for (let p = 0; p < 45; p++) {
          const col = riverPebbleColors[p % riverPebbleColors.length];
          const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.55 });
          const rx = 0.2 + (Math.sin(p * 2.3) * 0.15 + 0.15);
          const rz = 0.2 + (Math.cos(p * 3.1) * 0.15 + 0.15);
          const pMesh = new THREE.Mesh(new THREE.SphereGeometry(rx, 7, 6), mat);
          pMesh.scale.set(1.1, 0.45, 1.2);
          const px = -25.0 + ((p * 1.17) % 50.0);
          const pz = -8.5 + ((p * 2.31) % 17.0);
          pMesh.position.set(px, 0.08, pz);
          pMesh.rotation.y = p * 0.4;
          this.sceneryGroup.add(pMesh);
        }

        // 2. Aquascape Hardscape: ROCKS, DRIFTWOOD & NATURAL CAVES (No houses!)
        this.buildAquascapeRocks();

        // 3. Rectangular Glass Tank Framing (Real Aquarium Enclosure)
        const rimMat = new THREE.MeshStandardMaterial({
          color: 0x111827,
          metalness: 0.8,
          roughness: 0.25,
        });

        // Bottom base trim framing the sand bed
        const bRimFront = new THREE.Mesh(new THREE.BoxGeometry(TANK_WIDTH + 0.6, 0.6, 0.4), rimMat);
        bRimFront.position.set(0, 0.1, TANK_DEPTH / 2 + 0.1);
        const bRimBack = new THREE.Mesh(new THREE.BoxGeometry(TANK_WIDTH + 0.6, 0.6, 0.4), rimMat);
        bRimBack.position.set(0, 0.1, -TANK_DEPTH / 2 - 0.1);
        const bRimLeft = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.6, TANK_DEPTH + 0.6), rimMat);
        bRimLeft.position.set(-TANK_WIDTH / 2 - 0.1, 0.1, 0);
        const bRimRight = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.6, TANK_DEPTH + 0.6), rimMat);
        bRimRight.position.set(TANK_WIDTH / 2 + 0.1, 0.1, 0);
        this.sceneryGroup.add(bRimFront, bRimBack, bRimLeft, bRimRight);

        // Top tank canopy / hood frame
        const tRimFront = new THREE.Mesh(new THREE.BoxGeometry(TANK_WIDTH + 0.8, 0.8, 0.5), rimMat);
        tRimFront.position.set(0, TANK_HEIGHT, TANK_DEPTH / 2 + 0.15);
        const tRimBack = new THREE.Mesh(new THREE.BoxGeometry(TANK_WIDTH + 0.8, 0.8, 0.5), rimMat);
        tRimBack.position.set(0, TANK_HEIGHT, -TANK_DEPTH / 2 - 0.15);
        const tRimLeft = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.8, TANK_DEPTH + 0.8), rimMat);
        tRimLeft.position.set(-TANK_WIDTH / 2 - 0.15, TANK_HEIGHT, 0);
        const tRimRight = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.8, TANK_DEPTH + 0.8), rimMat);
        tRimRight.position.set(TANK_WIDTH / 2 + 0.15, TANK_HEIGHT, 0);
        this.sceneryGroup.add(tRimFront, tRimBack, tRimLeft, tRimRight);

        // Top LED Light bar inside the hood
        const ledMat = new THREE.MeshBasicMaterial({ color: 0xe0f7ff });
        const ledBar = new THREE.Mesh(new THREE.BoxGeometry(TANK_WIDTH - 2.0, 0.12, 0.6), ledMat);
        ledBar.position.set(0, TANK_HEIGHT - 0.25, 0);
        this.sceneryGroup.add(ledBar);

        // 4 corner silicone/metal glass posts
        for (const cx of [-TANK_WIDTH / 2, TANK_WIDTH / 2]) {
          for (const cz of [-TANK_DEPTH / 2, TANK_DEPTH / 2]) {
            const corner = new THREE.Mesh(new THREE.BoxGeometry(0.35, TANK_HEIGHT, 0.35), rimMat);
            corner.position.set(cx, TANK_HEIGHT / 2.0, cz);
            this.sceneryGroup.add(corner);
          }
        }

        // Back Wall: Deep oceanic backdrop panel
        const backMat = new THREE.MeshStandardMaterial({
          color: 0x061e33,
          roughness: 0.6,
        });
        const backPanel = new THREE.Mesh(new THREE.PlaneGeometry(TANK_WIDTH, TANK_HEIGHT), backMat);
        backPanel.position.set(0, TANK_HEIGHT / 2.0, -TANK_DEPTH / 2 + 0.05);
        this.sceneryGroup.add(backPanel);

        // Shimmering Water Surface at top
        const waterTopMat = new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.32,
          roughness: 0.15,
          metalness: 0.2,
        });
        const waterTop = new THREE.Mesh(new THREE.PlaneGeometry(TANK_WIDTH, TANK_DEPTH), waterTopMat);
        waterTop.rotation.x = Math.PI / 2;
        waterTop.position.set(0, WATER_LEVEL, 0);
        this.sceneryGroup.add(waterTop);

        // 4. Air Stone Bubbler & Ambient Bubbles
        const airStoneMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.9 });
        const airStone = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.45, 1.2, 10), airStoneMat);
        airStone.position.set(-20.0, 0.6, -5.0);
        this.sceneryGroup.add(airStone);

        const bubbleMat = new THREE.MeshStandardMaterial({
          color: 0xc7f9ff,
          transparent: true,
          opacity: 0.6,
          roughness: 0.1,
          metalness: 0.1,
        });
        const bubbleGeo = new THREE.SphereGeometry(1.0, 8, 8);

        for (let b = 0; b < 55; b++) {
          const bubble = new THREE.Mesh(bubbleGeo, bubbleMat);
          const bScale = 0.09 + Math.random() * 0.18;
          bubble.scale.set(bScale, bScale * 1.15, bScale);
          // Many bubbles rise right above the air stone column
          let bx, bz;
          if (b < 25) {
            bx = -20.0 + (Math.random() - 0.5) * 1.8;
            bz = -5.0 + (Math.random() - 0.5) * 1.8;
          } else {
            bx = -25.0 + Math.random() * 50.0;
            bz = -8.0 + Math.random() * 16.0;
          }
          const by = 0.6 + Math.random() * (WATER_LEVEL - 1.0);
          bubble.position.set(bx, by, bz);
          bubble.userData = {
            speed: 1.8 + Math.random() * 2.4,
            phase: Math.random() * Math.PI * 2.0,
            driftAmp: 0.15 + Math.random() * 0.35,
            isAirStone: b < 25,
          };
          this.sceneryGroup.add(bubble);
          this.ambientBubbles.push(bubble);
        }
      }

      this.scene.add(this.sceneryGroup);
      this.scene.add(this.territoryGroup);
      this.updateSkylineCount();
    }

    buildAquascapeRocks() {
      const rockColors = [0x57534e, 0x78716c, 0x44403c, 0x64748b, 0x3f3f46];

      const addRock = (px, py, pz, sx, sy, sz, color, ry = 0) => {
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.88 });
        const geo = new THREE.DodecahedronGeometry(1.0, 1);
        const m = new THREE.Mesh(geo, mat);
        m.scale.set(sx, sy, sz);
        m.position.set(px, py, pz);
        m.rotation.set(0.1, ry, 0.08);
        m.castShadow = true;
        m.receiveShadow = true;
        this.sceneryGroup.add(m);
        return m;
      };

      // ── Rock Formation 1: Center Slate Cave (Swim-through arch) ──
      // Left pillar of cave
      addRock(-2.6, 1.8, -1.8, 1.8, 3.2, 1.8, rockColors[0], 0.2);
      addRock(-2.3, 3.8, -1.6, 1.4, 2.0, 1.5, rockColors[1], 0.5);
      // Right pillar of cave
      addRock(2.6, 1.8, -1.8, 1.8, 3.2, 1.8, rockColors[2], -0.2);
      addRock(2.3, 3.8, -1.6, 1.4, 2.0, 1.5, rockColors[0], -0.4);
      // Archway roof slab spanning the top (cave opening is ~3.5 units wide and 3.5 units tall)
      const roofSlab = addRock(0, 4.6, -1.6, 3.8, 1.2, 2.2, rockColors[1], 0.1);
      roofSlab.rotation.z = -0.05;
      // Green moss growing on top of the cave roof
      const mossMat = new THREE.MeshStandardMaterial({ color: 0x3f6212, roughness: 0.95 });
      const moss = new THREE.Mesh(new THREE.SphereGeometry(1.0, 8, 6), mossMat);
      moss.scale.set(3.2, 0.4, 1.6);
      moss.position.set(0, 5.3, -1.6);
      this.sceneryGroup.add(moss);

      // ── Rock Formation 2: Left Mountain / Slate Stack ──
      addRock(-16.0, 2.2, -3.5, 3.2, 4.0, 2.8, rockColors[2], 0.3);
      addRock(-15.5, 5.8, -3.2, 2.4, 3.6, 2.2, rockColors[0], -0.2);
      addRock(-15.0, 9.2, -3.0, 1.8, 3.2, 1.8, rockColors[1], 0.6);
      addRock(-14.8, 11.8, -2.8, 1.2, 2.4, 1.2, rockColors[3], 0.1);

      // ── Rock Formation 3: Right Dragon Stone Ridge ──
      addRock(15.5, 2.5, -3.0, 3.0, 4.6, 2.8, rockColors[1], -0.4);
      addRock(15.0, 6.4, -2.8, 2.2, 4.0, 2.2, rockColors[3], 0.2);
      addRock(14.5, 9.8, -2.6, 1.5, 3.2, 1.6, rockColors[0], -0.1);

      // ── Spiderwood & Driftwood Roots (Curved branches) ──
      const woodMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.85 });

      // Left curved branch extending forward
      const root1 = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.42, 6.2, 8), woodMat);
      root1.position.set(-8.0, 2.6, 1.8);
      root1.rotation.set(0.6, 0.4, 0.75);
      this.sceneryGroup.add(root1);
      const twig1 = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, 3.5, 6), woodMat);
      twig1.position.set(-6.8, 4.5, 1.2);
      twig1.rotation.set(0.2, -0.4, -0.6);
      this.sceneryGroup.add(twig1);

      // Right driftwood branch arching upwards
      const root2 = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.38, 5.8, 8), woodMat);
      root2.position.set(8.5, 2.8, 1.5);
      root2.rotation.set(-0.5, -0.3, -0.65);
      this.sceneryGroup.add(root2);

      // ── Live Aquatic Plants (Tall Vallisneria ribbons & Anubias leaves) ──
      const kelpMat = new THREE.MeshStandardMaterial({
        color: 0x10b981,
        roughness: 0.45,
        side: THREE.DoubleSide,
      });

      // Tall Vallisneria Eelgrass along the right back corner (reaches high in the water!)
      const plantGroupRight = new THREE.Group();
      plantGroupRight.position.set(21.0, 0, -4.5);
      for (let k = 0; k < 6; k++) {
        const h = 10.0 + k * 1.5;
        const ribbon = new THREE.Mesh(new THREE.BoxGeometry(0.4, h, 0.05), kelpMat);
        ribbon.position.set((k - 2.5) * 0.6, h / 2.0, (Math.sin(k) * 0.8));
        ribbon.rotation.set(0.1, k * 0.4, (k - 2.5) * 0.06);
        plantGroupRight.add(ribbon);
      }
      this.sceneryGroup.add(plantGroupRight);
      plantGroupRight.userData = { phase: 0.5 };
      this.swayingPlants.push(plantGroupRight);

      // Midground plant grove near left slate stack
      const plantGroupLeft = new THREE.Group();
      plantGroupLeft.position.set(-11.5, 0, -2.5);
      for (let k = 0; k < 5; k++) {
        const h = 7.5 + k * 1.2;
        const ribbon = new THREE.Mesh(new THREE.BoxGeometry(0.35, h, 0.05), kelpMat);
        ribbon.position.set((k - 2) * 0.5, h / 2.0, Math.cos(k) * 0.6);
        ribbon.rotation.set(0.08, k * 0.5, (k - 2) * 0.07);
        plantGroupLeft.add(ribbon);
      }
      this.sceneryGroup.add(plantGroupLeft);
      plantGroupLeft.userData = { phase: 1.8 };
      this.swayingPlants.push(plantGroupLeft);

      // Broad Anubias leaves clustered on the rock crevices
      const anubiasMat = new THREE.MeshStandardMaterial({ color: 0x059669, roughness: 0.5 });
      const leafGeo = new THREE.SphereGeometry(0.65, 8, 6);
      leafGeo.scale(1.0, 0.15, 1.6);
      for (let a = 0; a < 6; a++) {
        const leaf = new THREE.Mesh(leafGeo, anubiasMat);
        leaf.position.set(-2.0 + (a % 3) * 1.8, 4.8 + Math.floor(a / 3) * 0.8, -1.4 + (a % 2) * 0.5);
        leaf.rotation.set(0.35, a * 1.0, 0.2);
        this.sceneryGroup.add(leaf);
      }
    }

    createSuburbanLandmark(place) {
      const Point = place.pos;
      const tint = (place.tint[0] << 16) | (place.tint[1] << 8) | place.tint[2];
      const meshes = [];

      if (place.kind === Kinds.Business) {
        const shaftMat = new THREE.MeshStandardMaterial({ color: tint, roughness: 0.7 });
        const shaft = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.0, 1.6), shaftMat);
        shaft.castShadow = true;
        shaft.receiveShadow = true;

        const roofMat = new THREE.MeshStandardMaterial({ color: 0x5a5c62, roughness: 0.6 });
        const roof = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.3, 1.8), roofMat);
        roof.castShadow = true;

        const antennaMat = new THREE.MeshStandardMaterial({
          color: 0x3c3e44,
          metalness: 0.6,
          roughness: 0.4,
        });
        const antenna = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.0, 0.08), antennaMat);
        antenna.castShadow = true;

        const winMat = new THREE.MeshStandardMaterial({
          color: 0x9dcdde,
          emissive: 0x224455,
          roughness: 0.2,
        });
        const windowBand = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.25, 0.04), winMat);

        this.sceneryGroup.add(shaft, roof, antenna, windowBand);
        meshes.push(shaft, roof, antenna, windowBand);
      } else if (place.kind === Kinds.Church) {
        const naveMat = new THREE.MeshStandardMaterial({ color: tint, roughness: 0.8 });
        const nave = new THREE.Mesh(new THREE.BoxGeometry(2.0, 3.0, 1.8), naveMat);
        nave.castShadow = true;

        const roofMat = new THREE.MeshStandardMaterial({ color: 0x6e4a41, roughness: 0.7 });
        const roof = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.4, 2.0), roofMat);
        roof.castShadow = true;

        const spireMat = new THREE.MeshStandardMaterial({ color: 0xd6cdbe, roughness: 0.5 });
        const spire = new THREE.Mesh(new THREE.ConeGeometry(0.85, 1.0, 12), spireMat);
        spire.castShadow = true;

        const finialMat = new THREE.MeshStandardMaterial({
          color: 0xdeb85c,
          metalness: 0.8,
          roughness: 0.2,
        });
        const finial = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.3, 0.1), finialMat);

        this.sceneryGroup.add(nave, roof, spire, finial);
        meshes.push(nave, roof, spire, finial);
      } else if (place.kind === Kinds.Telescope) {
        const drumMat = new THREE.MeshStandardMaterial({ color: 0xa8a8a3, roughness: 0.7 });
        const drum = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 1.6, 16), drumMat);
        drum.castShadow = true;

        const domeMat = new THREE.MeshStandardMaterial({ color: 0xc8c8c3, roughness: 0.5 });
        const dome = new THREE.Mesh(
          new THREE.SphereGeometry(1.1, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.5),
          domeMat
        );
        dome.castShadow = true;

        const barrelMat = new THREE.MeshStandardMaterial({
          color: 0x4285c7,
          roughness: 0.4,
          metalness: 0.5,
        });
        const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 1.0, 12), barrelMat);
        barrel.castShadow = true;

        this.sceneryGroup.add(drum, dome, barrel);
        meshes.push(drum, dome, barrel);
      }

      const lm = { place, kind: place.kind, pos: Point, meshes };
      this.growLandmark(lm, place.activityCount);
      this.landmarks.push(lm);
    }

    growLandmark(lm, tier) {
      if (this.theme !== 'Suburban') return;
      const t = Math.min(MAX_BUILD_TIER, tier);
      const { x, z } = lm.pos;

      if (lm.kind === Kinds.Business) {
        const height = 3.0 + t * 0.85;
        const shaft = lm.meshes[0];
        shaft.scale.set(1.0, height, 1.0);
        shaft.position.set(x, height / 2.0, z);

        const roof = lm.meshes[1];
        roof.position.set(x, height + 0.15, z);

        const antenna = 0.8 + t * 0.04;
        const antMesh = lm.meshes[2];
        antMesh.scale.set(1.0, antenna, 1.0);
        antMesh.position.set(x, height + 0.3 + antenna / 2.0, z);

        const winBand = lm.meshes[3];
        winBand.position.set(x, height * 0.6, z + 0.82);
      } else if (lm.kind === Kinds.Church) {
        const nave = lm.meshes[0];
        nave.position.set(x, 1.5, z);

        const roof = lm.meshes[1];
        roof.position.set(x, 3.2, z);

        const spireHeight = 1.5 + t * 0.5;
        const spire = lm.meshes[2];
        spire.scale.set(1.0, spireHeight, 1.0);
        spire.position.set(x, 3.4 + spireHeight / 2.0, z);

        const finial = lm.meshes[3];
        finial.position.set(x, 3.4 + spireHeight + 0.15, z);
      } else if (lm.kind === Kinds.Telescope) {
        const drum = lm.meshes[0];
        drum.position.set(x, 0.8, z);

        const dome = lm.meshes[1];
        dome.position.set(x, 1.6, z);

        const barrelLength = 1.2 + t * 0.22;
        const barrel = lm.meshes[2];
        barrel.scale.set(1.0, barrelLength, 1.0);
        barrel.position.set(x, 2.0, z + 0.4);
        barrel.rotation.set(0.65, 0, 0);
      }
    }

    createSuburbanSceneryItem(p) {
      const x = p.pos.x;
      const z = p.pos.z;
      const tint = (p.tint[0] << 16) | (p.tint[1] << 8) | p.tint[2];

      const addBox = (px, py, pz, sx, sy, sz, color, emissive = 0) => {
        const mat = new THREE.MeshStandardMaterial({
          color,
          roughness: 0.8,
          emissive,
        });
        const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
        m.position.set(px, py, pz);
        m.castShadow = true;
        m.receiveShadow = true;
        this.sceneryGroup.add(m);
        return m;
      };

      const addSphere = (px, py, pz, rx, ry, rz, color, emissive = 0) => {
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.7, emissive });
        const m = new THREE.Mesh(new THREE.SphereGeometry(1.0, 12, 10), mat);
        m.scale.set(rx, ry, rz);
        m.position.set(px, py, pz);
        m.castShadow = true;
        this.sceneryGroup.add(m);
        return m;
      };

      if (p.kind === Kinds.House) {
        addBox(x, 1.4, z, 3.6, 2.8, 3.3, tint);
        addBox(x, 3.05, z, 4.1, 0.5, 3.8, 0x6e4a41);
        addBox(x, 0.65, z + 1.67, 0.65, 1.3, 0.1, 0x5b4231);
        addBox(x - 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.0, 3.4, z - 0.7, 0.45, 1.0, 0.5, 0x905542);
      } else if (p.kind === Kinds.Shop) {
        addBox(x, 1.1, z, 3.6, 2.2, 3.3, tint);
        addBox(x, 2.45, z, 4.1, 0.5, 3.8, 0x6e4a41);
        addBox(x, 0.65, z + 1.67, 0.65, 1.3, 0.1, 0x5b4231);
        addBox(x - 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x, 2.0, z + 1.9, 3.8, 0.25, 0.8, 0xe39d4e);
      } else if (p.kind === Kinds.Tree) {
        addBox(x, 0.85, z, 0.4, 1.7, 0.4, 0x705133);
        addSphere(x, 2.1, z, 1.05, 1.2, 1.05, 0x3b773e);
      } else if (p.kind === Kinds.Bench) {
        addBox(x, 0.45, z, 1.4, 0.15, 0.6, 0x99693e);
        addBox(x, 0.8, z - 0.25, 1.4, 0.65, 0.12, 0x99693e);
        addBox(x - 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
        addBox(x + 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
      } else if (p.kind === Kinds.Ball) {
        addSphere(x, 0.3, z, 0.3, 0.3, 0.3, 0xf17951);
      } else if (p.kind === Kinds.Garden) {
        addBox(x, 0.09, z, 0.9, 0.18, 0.9, 0x6a4d31);
        for (const offset of [-0.25, 0.0, 0.25]) {
          addSphere(x + offset, 0.22, z, 0.1, 0.175, 0.32, 0x559b3e);
        }
      } else if (p.kind === Kinds.Flowers) {
        const flowerColors = [0xef799d, 0xf8cb4f, 0xa886d2];
        for (let idx = 0; idx < flowerColors.length; idx++) {
          const off = (idx - 1.0) * 0.23;
          addSphere(x + off, 0.25, z, 0.12, 0.15, 0.15, flowerColors[idx]);
        }
      } else if (p.kind === Kinds.Mailbox) {
        addBox(x, 0.5, z, 0.12, 1.0, 0.12, 0x605549);
        addBox(x, 1.05, z, 0.6, 0.4, 0.45, 0x49739b);
      } else if (p.kind === Kinds.Pond) {
        addSphere(x, 0.02, z, 1.6, 0.075, 1.6, 0x58a5b8);
      }
    }

    initCitizens() {
      for (const y of this.yumons) {
        if (y.mesh) {
          disposeHierarchy(y.mesh);
          this.scene.remove(y.mesh);
        }
      }
      this.yumons = [];

      for (let i = 0; i < this.citizens.length; i++) {
        const c = this.citizens[i];
        const group = new THREE.Group();

        // Yumon stylized pet mesh
        // Body: soft rounded capsule/egg shape
        const bodyMat = new THREE.MeshStandardMaterial({
          color: c.color,
          roughness: 0.4,
          metalness: 0.1,
        });
        const bodyGeo = new THREE.SphereGeometry(0.55, 16, 14);
        bodyGeo.scale(1.0, 1.2, 1.0);
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = 0.65;
        body.castShadow = true;
        group.add(body);

        // Belly patch
        const bellyMat = new THREE.MeshStandardMaterial({
          color: c.secondaryColor,
          roughness: 0.6,
        });
        const belly = new THREE.Mesh(new THREE.SphereGeometry(0.38, 12, 10), bellyMat);
        belly.position.set(0, 0.58, 0.32);
        belly.scale.set(0.85, 1.0, 0.4);
        group.add(belly);

        // Cute Ears
        const earMat = new THREE.MeshStandardMaterial({
          color: c.color,
          roughness: 0.4,
        });
        for (const side of [-0.28, 0.28]) {
          const ear = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.4, 8), earMat);
          ear.position.set(side, 1.25, 0.05);
          ear.rotation.set(-0.15, 0, side > 0 ? -0.3 : 0.3);
          ear.castShadow = true;
          group.add(ear);
        }

        // Eyes
        const eyeMat = new THREE.MeshBasicMaterial({ color: 0x1a1a24 });
        const pupilMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        for (const side of [-0.18, 0.18]) {
          const eye = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8), eyeMat);
          eye.position.set(side, 0.78, 0.48);
          eye.scale.set(1.0, 1.3, 0.5);
          group.add(eye);

          const glint = new THREE.Mesh(new THREE.SphereGeometry(0.03, 6, 6), pupilMat);
          glint.position.set(side + 0.02, 0.81, 0.52);
          group.add(glint);
        }

        // Cheeks
        const cheekMat = new THREE.MeshBasicMaterial({ color: 0xff8fa3 });
        for (const side of [-0.3, 0.3]) {
          const cheek = new THREE.Mesh(new THREE.SphereGeometry(0.07, 6, 6), cheekMat);
          cheek.position.set(side, 0.66, 0.46);
          cheek.scale.set(1.0, 0.8, 0.4);
          group.add(cheek);
        }

        // Little Feet/Paws
        const pawMat = new THREE.MeshStandardMaterial({
          color: c.secondaryColor,
          roughness: 0.5,
        });
        const leftPaw = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), pawMat);
        leftPaw.position.set(-0.25, 0.12, 0.1);
        leftPaw.scale.set(1.0, 0.7, 1.4);
        group.add(leftPaw);

        const rightPaw = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), pawMat);
        rightPaw.position.set(0.25, 0.12, 0.1);
        rightPaw.scale.set(1.0, 0.7, 1.4);
        group.add(rightPaw);

        // Ground shadow blob
        const shadowMat = new THREE.MeshBasicMaterial({
          color: 0x000000,
          transparent: true,
          opacity: 0.28,
        });
        const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.5, 12), shadowMat);
        shadow.rotation.x = -Math.PI / 2;
        shadow.position.y = 0.02;
        group.add(shadow);

        // Cute Scuba Goggles & Snorkel for Fish Tank
        const scubaGroup = new THREE.Group();
        const goggleFrameMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.3 });
        const goggleLensMat = new THREE.MeshBasicMaterial({ color: 0xa5f3fc, transparent: true, opacity: 0.45 });
        for (const side of [-0.18, 0.18]) {
          const rim = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.022, 6, 12), goggleFrameMat);
          rim.position.set(side, 0.78, 0.5);
          scubaGroup.add(rim);
          const lens = new THREE.Mesh(new THREE.CircleGeometry(0.1, 8), goggleLensMat);
          lens.position.set(side, 0.78, 0.5);
          scubaGroup.add(lens);
        }
        const strap = new THREE.Mesh(new THREE.TorusGeometry(0.56, 0.02, 6, 14), goggleFrameMat);
        strap.rotation.x = Math.PI / 2;
        strap.position.set(0, 0.78, 0.05);
        scubaGroup.add(strap);
        const snorkelMat = new THREE.MeshStandardMaterial({ color: 0xf43f5e, roughness: 0.4 });
        const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.65, 8), snorkelMat);
        tube.position.set(0.42, 1.05, 0.12);
        tube.rotation.z = -0.22;
        tube.rotation.x = 0.15;
        scubaGroup.add(tube);
        const snorkelCap = new THREE.Mesh(new THREE.SphereGeometry(0.055, 6, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }));
        snorkelCap.position.set(0.49, 1.4, 0.08);
        scubaGroup.add(snorkelCap);

        scubaGroup.visible = this.theme === 'FishTank';
        group.add(scubaGroup);

        // Initial coordinates based on theme
        let px, py, pz;
        if (this.theme === 'FishTank') {
          px = -14.0 + i * 9.0;
          py = 4.0 + i * 3.5; // Natural 3D swimming heights
          pz = (i % 2 === 0 ? -2.0 : 2.0);
        } else {
          const col = c.homeParcel % 4;
          const row = Math.floor(c.homeParcel / 4);
          px = -22.5 + col * 15.0 - 4.5 + (i % 2) * 2.5;
          py = 0.45;
          pz = -22.5 + row * 15.0 - 4.5 + Math.floor(i / 2) * 2.5;
        }

        group.position.set(px, py, pz);
        group.scale.set(0.85, 0.85, 0.85);
        this.scene.add(group);

        this.yumons.push({
          id: i,
          citizen: c,
          mesh: group,
          scubaGroup,
          shadow,
          pos: { x: px, y: py, z: pz },
          target: { x: px, y: py, z: pz },
          behavior: Behaviors.Rest,
          targetPlace: null,
          targetFlake: null,
          facing: 0,
          pitch: 0,
          arrived: true,
          nextDecisionTime: performance.now() + i * 2000,
          travelSeconds: 0,
          leftPaw,
          rightPaw,
          body,
        });
      }
    }

    initHUD() {
      this.hud = document.createElement('div');
      this.hud.className = 'universe-hud';
      this.hud.innerHTML = `
        <div class="hud-top">
          <div class="hud-badge"><span class="hud-pulse"></span><strong>Yumon Universe</strong></div>
          <button class="hud-btn" id="theme-btn" title="Toggle Suburban / Fish Tank view">Suburban 🌳</button>
          <button class="hud-btn" id="feed-btn" title="Drop delicious fish food flakes" style="display: none;">Feed 🦐</button>
          <div class="hud-skyline" id="hud-skyline" title="Current world status">🏗️ Skyline: Lv. 0</div>
          <div class="hud-actions">
            <button class="hud-icon-btn" id="cam-reset-btn" title="Reset Camera View">🎥</button>
            <button class="hud-icon-btn" id="pause-btn" title="Pause / Play simulation">⏸️</button>
          </div>
        </div>
        <div class="hud-citizen-card" id="hud-citizen-card">
          <div class="hud-card-avatar" id="hud-card-avatar">🦊</div>
          <div class="hud-card-body">
            <div class="hud-card-header"><strong id="hud-card-name">Ember</strong> <span id="hud-card-role">Architect</span></div>
            <div class="hud-card-status" id="hud-card-status">Resting peacefully</div>
          </div>
          <button class="hud-chat-btn" id="hud-chat-btn" title="Chat with this Yumon">Chat 💬</button>
        </div>
      `;
      this.container.appendChild(this.hud);

      // HUD Button Events
      const themeBtn = this.hud.querySelector('#theme-btn');
      const feedBtn = this.hud.querySelector('#feed-btn');

      themeBtn.addEventListener('click', () => {
        this.theme = this.theme === 'Suburban' ? 'FishTank' : 'Suburban';
        themeBtn.textContent = this.theme === 'Suburban' ? 'Suburban 🌳' : 'Fish Tank 🐠';
        feedBtn.style.display = this.theme === 'FishTank' ? 'inline-block' : 'none';

        this.setCameraForTheme(this.theme);
        this.updateSkyAndFog();
        this.buildWorld();

        // Reposition Yumons for active theme
        for (let i = 0; i < this.yumons.length; i++) {
          const y = this.yumons[i];
          if (this.theme === 'FishTank') {
            y.pos = { x: -14.0 + i * 9.0, y: 4.0 + i * 3.5, z: (i % 2 === 0 ? -2.0 : 2.0) };
            y.target = { ...y.pos };
            y.behavior = Behaviors.Explore;
          } else {
            const col = y.citizen.homeParcel % 4;
            const row = Math.floor(y.citizen.homeParcel / 4);
            y.pos = {
              x: -22.5 + col * 15.0 - 4.5 + (i % 2) * 2.5,
              y: 0.45,
              z: -22.5 + row * 15.0 - 4.5 + Math.floor(i / 2) * 2.5,
            };
            y.target = { ...y.pos };
            y.behavior = Behaviors.Rest;
          }
          y.arrived = true;
          y.targetPlace = null;
          y.targetFlake = null;
          y.mesh.position.set(y.pos.x, y.pos.y, y.pos.z);
          if (y.scubaGroup) {
            y.scubaGroup.visible = this.theme === 'FishTank';
          }
        }

        this.updateHUDCard();
      });

      feedBtn.addEventListener('click', () => {
        this.dropFood(
          (Math.random() - 0.5) * 16.0,
          (Math.random() - 0.5) * 6.0
        );
      });

      const camResetBtn = this.hud.querySelector('#cam-reset-btn');
      camResetBtn.addEventListener('click', () => {
        this.setCameraForTheme(this.theme);
      });

      const pauseBtn = this.hud.querySelector('#pause-btn');
      pauseBtn.addEventListener('click', () => {
        this.paused = !this.paused;
        pauseBtn.textContent = this.paused ? '▶️' : '⏸️';
      });

      const chatBtn = this.hud.querySelector('#hud-chat-btn');
      chatBtn.addEventListener('click', () => {
        const citizen = this.citizens[this.selectedCitizen];
        const chatTab = document.getElementById('chat-tab');
        if (chatTab) chatTab.click();
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
          if (this.theme === 'FishTank') {
            chatInput.value = `Hi ${citizen.name}! What are you up to in the aquarium today? `;
          } else {
            chatInput.value = `Hi ${citizen.name}! What are you building today? `;
          }
          chatInput.focus();
        }
      });

      this.updateHUDCard();
    }

    // Interactive Feeding: drops 6-8 multi-colored fish food flakes from fixed pool
    dropFood(centerX, centerZ) {
      if (this.theme !== 'FishTank') return;

      const count = 6 + Math.floor(Math.random() * 3);

      for (let f = 0; f < count; f++) {
        const flakeMesh = this.foodFlakes[this.flakePoolIndex];
        this.flakePoolIndex = (this.flakePoolIndex + 1) % this.flakePoolSize;

        const mat = this.flakeMaterials[Math.floor(Math.random() * this.flakeMaterials.length)];
        flakeMesh.material = mat;

        const fx = Math.max(-24.0, Math.min(24.0, centerX + (Math.random() - 0.5) * 6.0));
        const fz = Math.max(-7.0, Math.min(7.0, centerZ + (Math.random() - 0.5) * 4.0));
        const fy = WATER_LEVEL - 0.2;

        flakeMesh.position.set(fx, fy, fz);
        flakeMesh.rotation.set(Math.random(), Math.random(), Math.random());

        const u = flakeMesh.userData;
        u.active = true;
        u.eaten = false;
        u.pos.x = fx;
        u.pos.y = fy;
        u.pos.z = fz;
        u.vy = 0.9 + Math.random() * 0.9;
        u.phase = Math.random() * Math.PI * 2.0;
        u.rotSpeed = (Math.random() - 0.5) * 4.0;
        u.settledTime = 0;
        flakeMesh.visible = true;

        // Little feeding splash ripples on the water surface
        this.spawnSparkles(fx, WATER_LEVEL, fz, 0xbae6fd);
      }

      // Alert all Yumons that food has dropped!
      for (const y of this.yumons) {
        y.behavior = Behaviors.Eat;
        y.arrived = false;
        y.travelSeconds = 0;
        this.findNearestFoodFlake(y);
      }

      const statusEl = this.hud?.querySelector('#hud-card-status');
      if (statusEl) {
        statusEl.textContent = 'Dropped tasty fish flakes! 🦐 The Yumons are swimming to eat!';
      }
    }

    findNearestFoodFlake(y) {
      if (!this.foodFlakes || this.foodFlakes.length === 0) return null;
      let nearest = null;
      let minDist = Infinity;
      for (let i = 0; i < this.foodFlakes.length; i++) {
        const f = this.foodFlakes[i];
        const u = f.userData;
        if (!u.active || u.eaten) continue;
        const dist = Math.hypot(y.pos.x - u.pos.x, y.pos.y - u.pos.y, y.pos.z - u.pos.z);
        if (dist < minDist) {
          minDist = dist;
          nearest = u;
        }
      }
      if (nearest) {
        y.targetFlake = nearest;
        y.target = { x: nearest.pos.x, y: nearest.pos.y, z: nearest.pos.z };
      }
      return nearest;
    }

    updateSkylineCount() {
      const skylineEl = this.hud?.querySelector('#hud-skyline');
      if (!skylineEl) return;

      if (this.theme === 'FishTank') {
        skylineEl.textContent = `🦐 Flakes Fed: ${this.flakesEatenTotal}`;
      } else {
        let totalLevels = 0;
        for (const lm of this.landmarks) {
          totalLevels += lm.place.activityCount || 0;
        }
        skylineEl.textContent = `🏗️ Skyline: Lv. ${totalLevels}`;
      }
    }

    updateHUDCard() {
      const y = this.yumons[this.selectedCitizen];
      if (!y) return;
      const nameEl = this.hud?.querySelector('#hud-card-name');
      const roleEl = this.hud?.querySelector('#hud-card-role');
      const statusEl = this.hud?.querySelector('#hud-card-status');
      const avatarEl = this.hud?.querySelector('#hud-card-avatar');

      if (nameEl) nameEl.textContent = y.citizen.name;
      if (roleEl) {
        roleEl.textContent = this.theme === 'FishTank' ? y.citizen.fishRole : y.citizen.role;
      }
      if (statusEl) {
        const actionLabel = BehaviorLabels[this.theme][y.behavior] || y.behavior;
        if (this.theme === 'FishTank') {
          const targetNoun = y.targetPlace !== null && this.aquascapePlaces[y.targetPlace]
            ? this.aquascapePlaces[y.targetPlace].name
            : 'open aquarium water';
          statusEl.textContent = `${actionLabel} at the ${targetNoun} · Happy 🫧`;
        } else {
          const targetNoun = y.targetPlace !== null && this.world.places[y.targetPlace]
            ? Nouns[this.theme][this.world.places[y.targetPlace].kind]
            : 'neighborhood';
          statusEl.textContent = `${actionLabel} at the ${targetNoun} · ${y.citizen.savings} coins`;
        }
      }
      if (avatarEl) {
        const icons = this.theme === 'FishTank' ? ['🐠', '🐟', '🐡', '🐬'] : ['🦊', '🐬', '🌿', '✨'];
        avatarEl.textContent = icons[y.id % icons.length];
      }
    }

    onClick(e) {
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      // Check Yumon meshes first
      for (const y of this.yumons) {
        const intersects = this.raycaster.intersectObjects(y.mesh.children, true);
        if (intersects.length > 0) {
          this.selectedCitizen = y.id;
          this.updateHUDCard();
          this.spawnSparkles(y.pos.x, y.pos.y + 0.6, y.pos.z, y.citizen.color);
          return;
        }
      }

      if (this.theme === 'FishTank') {
        // In Fish Tank: Clicking inside the tank drops fish food flakes right there!
        if (this.raycaster.ray.intersectPlane(this.planeZ0, this.intersectPoint)) {
          const dropX = Math.max(-24.0, Math.min(24.0, this.intersectPoint.x));
          const dropZ = (Math.random() - 0.5) * 6.0;
          this.dropFood(dropX, dropZ);
        } else {
          this.dropFood(0, 0);
        }
        return;
      }

      // Check Suburban Landmarks
      for (const lm of this.landmarks) {
        const intersects = this.raycaster.intersectObjects(lm.meshes, true);
        if (intersects.length > 0) {
          const y = this.yumons[this.selectedCitizen];
          y.targetPlace = lm.place.id;
          y.behavior = Behaviors.Build;
          y.target = approachPlaceSuburban(y.pos, lm.place);
          y.arrived = false;
          y.travelSeconds = 0;
          this.updateHUDCard();
          this.spawnSparkles(lm.pos.x, 2.0, lm.pos.z, 0xffd166);
          return;
        }
      }
    }

    spawnSparkles(x, y, z, colorHex) {
      const isAquarium = this.theme === 'FishTank';
      const geo = isAquarium ? this.sparkleGeoAquarium : this.sparkleGeoSuburban;
      const mat = this.getSparkleMaterial(colorHex, isAquarium);

      for (let i = 0; i < 8; i++) {
        const p = this.sparklePool[this.sparklePoolIndex];
        this.sparklePoolIndex = (this.sparklePoolIndex + 1) % this.sparklePoolSize;

        p.geometry = geo;
        p.material = mat;
        p.position.set(
          x + (Math.random() - 0.5) * 0.8,
          y + (Math.random() - 0.5) * 0.8,
          z + (Math.random() - 0.5) * 0.8
        );
        p.scale.set(1, 1, 1);
        p.userData.vx = (Math.random() - 0.5) * 1.2;
        p.userData.vy = 0.8 + Math.random() * 1.5;
        p.userData.vz = (Math.random() - 0.5) * 1.2;
        p.userData.life = 1.0;
        p.userData.active = true;
        p.visible = true;
      }
    }

    tickSimulation(dt, now) {
      if (this.paused) return;

      const isAquarium = this.theme === 'FishTank';

      // Update sparkles from fixed pool
      for (let i = 0; i < this.sparklePoolSize; i++) {
        const p = this.sparklePool[i];
        if (!p.userData.active) continue;
        p.userData.life -= dt * 1.5;
        p.position.x += p.userData.vx * dt;
        p.position.y += p.userData.vy * dt;
        p.position.z += p.userData.vz * dt;
        p.scale.multiplyScalar(0.96);
        if (p.userData.life <= 0) {
          p.userData.active = false;
          p.visible = false;
        }
      }

      // Update rising ambient bubbles in Fish Tank mode
      if (isAquarium && this.ambientBubbles.length > 0) {
        for (let b = 0; b < this.ambientBubbles.length; b++) {
          const bubble = this.ambientBubbles[b];
          bubble.position.y += bubble.userData.speed * dt;
          bubble.position.x += Math.sin(now * 0.003 + bubble.userData.phase) * dt * bubble.userData.driftAmp;
          bubble.position.z += Math.cos(now * 0.0025 + bubble.userData.phase) * dt * bubble.userData.driftAmp;
          if (bubble.position.y >= WATER_LEVEL) {
            bubble.position.y = 0.5 + Math.random() * 0.4;
            if (bubble.userData.isAirStone) {
              bubble.position.x = -20.0 + (Math.random() - 0.5) * 1.4;
              bubble.position.z = -5.0 + (Math.random() - 0.5) * 1.4;
            } else {
              bubble.position.x = -24.0 + Math.random() * 48.0;
              bubble.position.z = -7.5 + Math.random() * 15.0;
            }
          }
        }
      }

      // Update swaying live plants
      if (isAquarium && this.swayingPlants.length > 0) {
        for (let sp = 0; sp < this.swayingPlants.length; sp++) {
          const plant = this.swayingPlants[sp];
          plant.rotation.z = Math.sin(now * 0.0018 + plant.userData.phase) * 0.12;
          plant.rotation.x = Math.cos(now * 0.0014 + plant.userData.phase) * 0.08;
        }
      }

      // Update Falling Food Flakes from fixed pool
      if (isAquarium && this.foodFlakes.length > 0) {
        for (let f = 0; f < this.foodFlakes.length; f++) {
          const flake = this.foodFlakes[f];
          const u = flake.userData;
          if (!u.active) continue;

          if (u.eaten) {
            u.active = false;
            flake.visible = false;
            continue;
          }

          if (u.pos.y > 0.4) {
            u.pos.y -= u.vy * dt;
            u.pos.x += Math.sin(now * 0.004 + u.phase) * dt * 0.5;
            u.pos.z += Math.cos(now * 0.003 + u.phase) * dt * 0.35;
            flake.rotation.x += u.rotSpeed * dt;
            flake.rotation.y += u.rotSpeed * 0.5 * dt;
            flake.position.set(u.pos.x, u.pos.y, u.pos.z);
          } else {
            // Flake has settled on the sand
            u.settledTime += dt;
            if (u.settledTime > 12.0) {
              u.active = false;
              flake.visible = false;
            }
          }
        }
      }

      // Tick Yumons
      for (let i = 0; i < this.yumons.length; i++) {
        const y = this.yumons[i];

        // Autonomous decision making
        if (y.arrived && now >= y.nextDecisionTime) {
          this.decideNextAction(y);
        }

        // If eating, update target to track falling flake
        if (isAquarium && y.behavior === Behaviors.Eat && y.targetFlake) {
          if (!y.targetFlake.eaten && y.targetFlake.active) {
            y.target.x = y.targetFlake.pos.x;
            y.target.y = y.targetFlake.pos.y;
            y.target.z = y.targetFlake.pos.z;
          } else {
            this.findNearestFoodFlake(y);
          }
        }

        if (isAquarium) {
          // ── FULL 3D VERTICAL SWIMMING MECHANICS (FISH TANK) ──
          const dx = y.target.x - y.pos.x;
          const dy = y.target.y - y.pos.y;
          const dz = y.target.z - y.pos.z;
          const dist3D = Math.hypot(dx, dy, dz);

          if (dist3D > 0.35) {
            y.travelSeconds += dt;
            if (y.travelSeconds > 15.0) {
              y.arrived = true;
              y.nextDecisionTime = now + 1500;
              continue;
            }

            const swimSpeed = (y.behavior === Behaviors.Eat ? 4.5 : 3.2) * dt;
            const step = Math.min(dist3D, swimSpeed);
            const dirX = dx / dist3D;
            const dirY = dy / dist3D;
            const dirZ = dz / dist3D;

            // 3D Position Update with tank boundary clamping
            y.pos.x = Math.max(-25.5, Math.min(25.5, y.pos.x + dirX * step));
            y.pos.y = Math.max(1.2, Math.min(WATER_LEVEL - 0.8, y.pos.y + dirY * step));
            y.pos.z = Math.max(-7.5, Math.min(7.5, y.pos.z + dirZ * step));

            // Horizontal facing
            y.facing = Math.atan2(dirX, dirZ);

            // Vertical pitch (pointing up when ascending, pointing down when diving)
            const horizDist = Math.hypot(dirX, dirZ);
            y.pitch = -Math.atan2(dirY, Math.max(0.01, horizDist));

            // Natural aquatic undulation & banking roll
            const swimStroke = Math.sin(now * 0.009 + y.id * 1.8);
            const waveY = Math.sin(now * 0.004 + y.id) * 0.2;

            y.mesh.position.set(y.pos.x, y.pos.y + waveY, y.pos.z);
            y.mesh.rotation.y = y.facing;
            y.mesh.rotation.x = y.pitch * 0.85 + Math.sin(now * 0.006 + y.id) * 0.08;
            y.mesh.rotation.z = Math.sin(now * 0.004 + y.id) * 0.16;

            // Flipper paddling
            if (y.leftPaw) y.leftPaw.position.z = Math.sin(now * 0.016 + y.id) * 0.18;
            if (y.rightPaw) y.rightPaw.position.z = -Math.sin(now * 0.016 + y.id) * 0.18;

            // Soft shadow follows below on the sand bed
            if (y.shadow) {
              y.shadow.position.y = 0.04 - y.mesh.position.y;
              const shadowScale = Math.max(0.3, 0.9 - y.mesh.position.y * 0.04);
              y.shadow.scale.set(shadowScale, shadowScale, shadowScale);
              y.shadow.material.opacity = Math.max(0.06, 0.25 - y.mesh.position.y * 0.012);
            }

            // Check if Yumon caught the food flake!
            if (y.behavior === Behaviors.Eat && y.targetFlake && !y.targetFlake.eaten && y.targetFlake.active) {
              const flakeDist = Math.hypot(
                y.pos.x - y.targetFlake.pos.x,
                y.pos.y - y.targetFlake.pos.y,
                y.pos.z - y.targetFlake.pos.z
              );
              if (flakeDist < 1.2) {
                // Nibble!
                y.targetFlake.eaten = true;
                this.flakesEatenTotal += 1;
                this.updateSkylineCount();
                this.spawnSparkles(y.pos.x, y.pos.y + 0.3, y.pos.z, 0xffd166);

                // Look for another food flake or finish eating
                const nextFlake = this.findNearestFoodFlake(y);
                if (!nextFlake) {
                  y.behavior = Behaviors.Play;
                  y.arrived = true;
                  y.nextDecisionTime = now + 4000;
                  this.updateHUDCard();
                }
              }
            }

            if (Math.random() < 0.03) {
              this.spawnSparkles(y.pos.x, y.pos.y, y.pos.z, 0x7dd3fc);
            }
          } else if (!y.arrived) {
            y.arrived = true;
            y.nextDecisionTime = now + 4000 + Math.random() * 3000;
            if (y.behavior === Behaviors.Eat) {
              const nextFlake = this.findNearestFoodFlake(y);
              if (!nextFlake) {
                y.behavior = Behaviors.Explore;
              }
            }
            if (y.id === this.selectedCitizen) this.updateHUDCard();
          } else {
            // Weightless floating & hovering in water column
            const hoverY = Math.sin(now * 0.0025 + y.id) * 0.16;
            y.mesh.position.set(y.pos.x, y.pos.y + hoverY, y.pos.z);
            y.mesh.rotation.x = Math.sin(now * 0.002) * 0.05;
            y.mesh.rotation.z = Math.sin(now * 0.0015 + y.id) * 0.08;
            const breath = 1.0 + Math.sin(now * 0.002) * 0.035;
            y.body.scale.set(1.0 * breath, 1.2 / breath, 1.0 * breath);
          }
        } else {
          // ── 2D WALKING MECHANICS (SUBURBAN) ──
          const dx = y.target.x - y.pos.x;
          const dz = y.target.z - y.pos.z;
          const dist = Math.hypot(dx, dz);

          if (dist > 0.08) {
            y.travelSeconds += dt;
            if (y.travelSeconds > 14.0) {
              y.target = y.pos;
              y.arrived = true;
              y.nextDecisionTime = now + 2000;
              continue;
            }

            const dirX = dx / dist;
            const dirZ = dz / dist;
            y.facing = Math.atan2(dirX, dirZ);

            const step = Math.min(dist, dt * 2.8);
            const proposed = { x: y.pos.x + dirX * step, z: y.pos.z + dirZ * step };

            if (isWalkableSuburban(proposed, this.world.places)) {
              y.pos = proposed;
            } else {
              const xOnly = { x: proposed.x, z: y.pos.z };
              const zOnly = { x: y.pos.x, z: proposed.z };
              if (Math.abs(dirX) > 0.01 && isWalkableSuburban(xOnly, this.world.places)) {
                y.pos = xOnly;
              } else if (Math.abs(dirZ) > 0.01 && isWalkableSuburban(zOnly, this.world.places)) {
                y.pos = zOnly;
              } else {
                y.target = y.pos;
                y.arrived = true;
                y.nextDecisionTime = now + 1500;
              }
            }

            const walkCycle = Math.sin(now * 0.01 + y.id * 2.0);
            y.mesh.position.set(y.pos.x, 0.45 + Math.abs(walkCycle) * 0.16, y.pos.z);
            y.mesh.rotation.y = y.facing;
            y.mesh.rotation.x = 0;
            y.mesh.rotation.z = Math.sin(now * 0.008 + y.id) * 0.08;
            if (y.leftPaw) y.leftPaw.position.z = walkCycle * 0.12;
            if (y.rightPaw) y.rightPaw.position.z = -walkCycle * 0.12;
            if (y.shadow) {
              y.shadow.position.y = 0.02;
              y.shadow.scale.set(1.0, 1.0, 1.0);
              y.shadow.material.opacity = 0.28;
            }
          } else if (!y.arrived) {
            y.arrived = true;
            this.onSuburbanArrived(y);
          } else {
            // Suburban idle animation
            if (y.behavior === Behaviors.Build) {
              const bounce = Math.abs(Math.sin(now * 0.008));
              y.mesh.position.set(y.pos.x, 0.45 + bounce * 0.28, y.pos.z);
              if (bounce > 0.9 && Math.random() < 0.25) {
                this.spawnSparkles(y.pos.x, 1.0, y.pos.z, 0xffd166);
              }
            } else if (y.behavior === Behaviors.Play) {
              const bounce = Math.abs(Math.sin(now * 0.006));
              y.mesh.position.set(y.pos.x, 0.45 + bounce * 0.2, y.pos.z);
            } else if (y.behavior === Behaviors.Rest) {
              y.mesh.position.set(y.pos.x, 0.38, y.pos.z);
              const breath = 1.0 + Math.sin(now * 0.003) * 0.04;
              y.body.scale.set(1.0 * breath, 1.2 / breath, 1.0 * breath);
            } else {
              y.mesh.position.set(y.pos.x, 0.45 + Math.sin(now * 0.003) * 0.03, y.pos.z);
            }
          }
        }
      }
    }

    onSuburbanArrived(y) {
      if (y.targetPlace !== null && this.world.places[y.targetPlace]) {
        const place = this.world.places[y.targetPlace];
        if (
          place.kind === Kinds.Business ||
          place.kind === Kinds.Church ||
          place.kind === Kinds.Telescope
        ) {
          place.activityCount += 1;
          const lm = this.landmarks.find((l) => l.place.id === place.id);
          if (lm) {
            this.growLandmark(lm, place.activityCount);
            this.updateSkylineCount();
            this.spawnSparkles(place.pos.x, 2.5, place.pos.z, 0xffd166);
          }
        }
      }
      y.nextDecisionTime = performance.now() + 6000 + Math.random() * 4000;
      if (y.id === this.selectedCitizen) this.updateHUDCard();
    }

    decideNextAction(y) {
      if (this.theme === 'FishTank') {
        // Natural Fish Tank Decisions across 3D water volume
        // If food flakes exist, prioritize food!
        if (this.foodFlakes.some((f) => f.userData.active && !f.userData.eaten) && Math.random() < 0.8) {
          const flake = this.findNearestFoodFlake(y);
          if (flake) {
            y.behavior = Behaviors.Eat;
            y.arrived = false;
            y.travelSeconds = 0;
            if (y.id === this.selectedCitizen) this.updateHUDCard();
            return;
          }
        }

        // Pick an aquascape feature (caves, dragon stones, plants, surface current)
        const choice = this.aquascapePlaces[Math.floor(Math.random() * this.aquascapePlaces.length)];
        const { behavior } = getVerbAndBehavior(choice.id, this.theme);
        y.behavior = behavior;
        y.targetPlace = choice.id;

        // Position with natural 3D swim variance around the feature
        const ox = (Math.random() - 0.5) * choice.radius;
        const oy = (Math.random() - 0.5) * 2.0;
        const oz = (Math.random() - 0.5) * (choice.radius * 0.7);

        y.target = {
          x: Math.max(-25.0, Math.min(25.0, choice.pos.x + ox)),
          y: Math.max(1.4, Math.min(WATER_LEVEL - 0.8, choice.pos.y + oy)),
          z: Math.max(-7.0, Math.min(7.0, choice.pos.z + oz)),
        };
        y.arrived = false;
        y.travelSeconds = 0;
      } else {
        // Suburban Decision Logic
        const nearby = [];
        for (let i = 0; i < this.world.places.length; i++) {
          const p = this.world.places[i];
          const dist = Math.hypot(y.pos.x - p.pos.x, y.pos.z - p.pos.z);
          if (dist <= SIGHT) {
            nearby.push({ id: i, place: p, dist });
          }
        }

        nearby.sort((a, b) => a.dist - b.dist);

        if (nearby.length > 0 && Math.random() < 0.85) {
          const choice = nearby[Math.floor(Math.random() * Math.min(5, nearby.length))];
          const { behavior } = getVerbAndBehavior(choice.place.kind, this.theme);
          y.behavior = behavior;
          y.targetPlace = choice.id;
          y.target = approachPlaceSuburban(y.pos, choice.place);
          y.arrived = false;
          y.travelSeconds = 0;
        } else {
          y.behavior = Behaviors.Explore;
          y.targetPlace = null;
          const angle = Math.random() * Math.PI * 2.0;
          const dist = 4.0 + Math.random() * 6.0;
          const target = {
            x: Math.max(-28, Math.min(28, y.pos.x + Math.cos(angle) * dist)),
            z: Math.max(-28, Math.min(28, y.pos.z + Math.sin(angle) * dist)),
          };
          if (isWalkableSuburban(target, this.world.places)) {
            y.target = target;
            y.arrived = false;
            y.travelSeconds = 0;
          }
        }
      }

      if (y.id === this.selectedCitizen) {
        this.updateHUDCard();
      }
    }

    // Called when RSS Chat or Products update
    onRssStateUpdate(state) {
      if (state.chat_busy) {
        const activeYumon = this.yumons[this.selectedCitizen];
        if (activeYumon) {
          activeYumon.behavior = Behaviors.Inspect;
          this.spawnSparkles(activeYumon.pos.x, activeYumon.pos.y + 0.8, activeYumon.pos.z, 0x9dd8b3);
          if (this.hud?.querySelector('#hud-card-status')) {
            const thinkingText = this.theme === 'FishTank'
              ? 'Bubbling over your idea... 💭'
              : 'Thinking about your idea... 💭';
            this.hud.querySelector('#hud-card-status').textContent = thinkingText;
          }
        }
      } else if (state.chat && state.chat.length > 0) {
        const lastMsg = state.chat[state.chat.length - 1];
        if (lastMsg.role === 'yumon') {
          const activeYumon = this.yumons[this.selectedCitizen];
          if (activeYumon) {
            this.spawnSparkles(activeYumon.pos.x, activeYumon.pos.y + 1.0, activeYumon.pos.z, 0xf8cb4f);
          }
        }
      }
    }

    setPainting(isPainting) {
      this.isPainting = isPainting;
      if (this.renderer?.shadowMap) {
        this.renderer.shadowMap.autoUpdate = !isPainting;
        if (!isPainting) {
          this.renderer.shadowMap.needsUpdate = true;
        }
      }
    }

    animate() {
      this.animId = requestAnimationFrame(this.animate);
      const now = performance.now();
      const targetFps = this.isPainting ? 10 : 30;
      const minInterval = 1000 / targetFps;
      const elapsed = now - (this.lastRenderTime || 0);
      if (elapsed < minInterval) {
        return;
      }
      this.lastRenderTime = now - (elapsed % minInterval);

      const dt = Math.min(0.1, (now - (this.lastTime || now)) / 1000.0);
      this.lastTime = now;

      this.tickSimulation(dt, now);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    }

    dispose() {
      if (this.animId) {
        cancelAnimationFrame(this.animId);
        this.animId = null;
      }
      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
      }
      if (this.sceneryGroup) {
        disposeHierarchy(this.sceneryGroup);
        this.scene.remove(this.sceneryGroup);
      }
      if (this.territoryGroup) {
        disposeHierarchy(this.territoryGroup);
        this.scene.remove(this.territoryGroup);
      }
      if (this.sparklesGroup) {
        disposeHierarchy(this.sparklesGroup);
        this.scene.remove(this.sparklesGroup);
      }
      if (this.flakesGroup) {
        disposeHierarchy(this.flakesGroup);
        this.scene.remove(this.flakesGroup);
      }
      for (const y of this.yumons) {
        if (y.mesh) {
          disposeHierarchy(y.mesh);
          this.scene.remove(y.mesh);
        }
      }
      if (this.sparkleMaterialCache) {
        this.sparkleMaterialCache.forEach((mat) => mat.dispose());
        this.sparkleMaterialCache.clear();
      }
      this.sparkleGeoAquarium?.dispose();
      this.sparkleGeoSuburban?.dispose();
      this.flakeGeo?.dispose();
      if (this.flakeMaterials) {
        this.flakeMaterials.forEach((mat) => mat.dispose());
      }
      this.renderer?.dispose();
    }
  }

  // Initialize on DOM load or immediately if already loaded
  function startUniverseApp() {
    if (window.__yumon_universe_app) return;
    const universeContainer = document.getElementById('universe-container');
    if (universeContainer) {
      window.__yumon_universe_app = new YumonUniverseApp(universeContainer);
    }
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', startUniverseApp);
  } else {
    startUniverseApp();
  }

  window.addEventListener('beforeunload', () => {
    if (window.__yumon_universe_app) {
      window.__yumon_universe_app.dispose();
      window.__yumon_universe_app = null;
    }
  });

  // Global callback for state sync
  window.__onUniverseRssSync = function (state) {
    if (window.__yumon_universe_app) {
      window.__yumon_universe_app.onRssStateUpdate(state);
    }
  };
})();
