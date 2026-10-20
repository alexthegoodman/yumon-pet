// Yumon Universe: Three.js Simulation World for Yumon RSS
(function () {
  'use strict';

  // Constants aligned with src/universe.rs
  const EXTENT = 30.0;
  const SIGHT = 12.0;
  const MAX_BUILD_TIER = 60;

  // PRNG matching universe.rs
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
    Business: 9, // Buildable Tower
    Church: 10,  // Buildable Temple/Spire
    Telescope: 11, // Buildable Observatory
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
    },
    Urban: {
      Rest: 'resting ☕',
      Visit: 'visiting 🏢',
      Play: 'reviewing product 📦',
      Collect: 'gathering money 💵',
      Tend: 'maintaining budget 📊',
      Inspect: 'looking around 🔍',
      Explore: 'exploring 🧭',
      Build: 'constructing tower 🔨',
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
    Urban: {
      [Kinds.House]: 'business',
      [Kinds.Shop]: 'market',
      [Kinds.Tree]: 'debt notice',
      [Kinds.Flowers]: 'money',
      [Kinds.Bench]: 'coffee counter',
      [Kinds.Ball]: 'product crate',
      [Kinds.Garden]: 'budget chart',
      [Kinds.Mailbox]: 'deal podium',
      [Kinds.Pond]: 'brand plaza',
      [Kinds.Business]: 'tower',
      [Kinds.Church]: 'temple',
      [Kinds.Telescope]: 'telescope',
    },
  };

  function getRadius(kind) {
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
    switch (kind) {
      case Kinds.Bench:
        return { verb: 'rest by', behavior: Behaviors.Rest };
      case Kinds.Ball:
        return { verb: 'play with', behavior: Behaviors.Play };
      case Kinds.Garden:
        return { verb: theme === 'Suburban' ? 'tend' : 'maintain', behavior: Behaviors.Tend };
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

  // Generate Neighborhood
  function generateNeighborhood(seed, theme) {
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

        const tint =
          theme === 'Suburban'
            ? [
                Math.floor(140 + random() * 90),
                Math.floor(140 + random() * 80),
                Math.floor(120 + random() * 90),
              ]
            : [
                Math.floor(90 + random() * 60),
                Math.floor(100 + random() * 65),
                Math.floor(120 + random() * 75),
              ];

        places.push({
          id: places.length,
          kind: mainKind,
          pos: { x: x + random() - 0.5, z: z + random() - 0.5 },
          radius: getRadius(mainKind),
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
            radius: getRadius(slotKind),
            tint,
            activityCount: 0,
          });
        }
      }
    }

    return { seed, theme, places };
  }

  function isWalkable(pos, places) {
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

  function approachPlace(from, place) {
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
      this.world = generateNeighborhood(this.seed, this.theme);
      this.paused = false;
      this.selectedCitizen = 0;

      this.landmarks = [];
      this.sceneryGroup = null;
      this.territoryGroup = null;
      this.sparklesGroup = null;

      this.citizens = [
        {
          name: 'Ember',
          kingdom: 0,
          color: 0xf27059,
          secondaryColor: 0xffd166,
          homeParcel: 0,
          role: 'Architect · Builder',
          savings: 42,
        },
        {
          name: 'Ripple',
          kingdom: 0,
          color: 0x4292f1,
          secondaryColor: 0x90e0ef,
          homeParcel: 1,
          role: 'Explorer · Scout',
          savings: 38,
        },
        {
          name: 'Fern',
          kingdom: 1,
          color: 0x48b376,
          secondaryColor: 0xb7e4c7,
          homeParcel: 3,
          role: 'Botanist · Planter',
          savings: 55,
        },
        {
          name: 'Sol',
          kingdom: 2,
          color: 0xf5a623,
          secondaryColor: 0xffe3a8,
          homeParcel: 14,
          role: 'Astronomer · Guide',
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
      this.updateSkyAndFog();

      this.camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 300);
      this.camera.position.set(38, 42, 48);
      this.camera.lookAt(0, 0, 0);

      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.container.appendChild(this.renderer.domElement);

      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 0, 0);
      this.controls.minDistance = 6;
      this.controls.maxDistance = 120;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;

      // Lights
      this.ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      this.scene.add(this.ambientLight);

      this.sunLight = new THREE.DirectionalLight(0xfff8ee, 1.2);
      this.sunLight.position.set(28, 48, 22);
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

      // Sparkles effect group
      this.sparklesGroup = new THREE.Group();
      this.scene.add(this.sparklesGroup);

      // Resize observer
      const resizeObserver = new ResizeObserver(() => {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w && h) {
          this.camera.aspect = w / h;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(w, h);
        }
      });
      resizeObserver.observe(this.container);

      // Raycaster for selecting Yumons or landmarks
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();
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

    updateSkyAndFog() {
      if (this.theme === 'Suburban') {
        const skyColor = new THREE.Color(0xa3c9e0);
        this.scene.background = skyColor;
        this.scene.fog = new THREE.FogExp2(0xa3c9e0, 0.007);
        if (this.hemiLight) this.hemiLight.groundColor.setHex(0x4a6350);
      } else {
        const skyColor = new THREE.Color(0x8fa3b5);
        this.scene.background = skyColor;
        this.scene.fog = new THREE.FogExp2(0x8fa3b5, 0.009);
        if (this.hemiLight) this.hemiLight.groundColor.setHex(0x3a424a);
      }
    }

    buildWorld() {
      if (this.sceneryGroup) this.scene.remove(this.sceneryGroup);
      if (this.territoryGroup) this.scene.remove(this.territoryGroup);

      this.sceneryGroup = new THREE.Group();
      this.territoryGroup = new THREE.Group();
      this.landmarks = [];

      const groundColor = this.theme === 'Suburban' ? 0x73a35b : 0xa8a8a3;
      const groundMat = new THREE.MeshStandardMaterial({
        color: groundColor,
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

        // Sidewalks
        const swX = new THREE.Mesh(new THREE.BoxGeometry(5.0, 0.05, 60.0), sidewalkMat);
        swX.position.set(axis, 0.01, 0);
        swX.receiveShadow = true;
        this.sceneryGroup.add(swX);

        const swZ = new THREE.Mesh(new THREE.BoxGeometry(60.0, 0.05, 5.0), sidewalkMat);
        swZ.position.set(0, 0.01, axis);
        swZ.receiveShadow = true;
        this.sceneryGroup.add(swZ);

        // Asphalt roads
        const rdX = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.05, 60.0), roadMat);
        rdX.position.set(axis, 0.035, 0);
        rdX.receiveShadow = true;
        this.sceneryGroup.add(rdX);

        const rdZ = new THREE.Mesh(new THREE.BoxGeometry(60.0, 0.05, 3.2), roadMat);
        rdZ.position.set(0, 0.035, axis);
        rdZ.receiveShadow = true;
        this.sceneryGroup.add(rdZ);

        // Road stripes
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

      // Territory Borders
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

      // Populate Scenery and Buildable Landmarks
      for (let i = 0; i < this.world.places.length; i++) {
        const place = this.world.places[i];
        if (
          place.kind === Kinds.Business ||
          place.kind === Kinds.Church ||
          place.kind === Kinds.Telescope
        ) {
          this.createLandmark(place);
        } else {
          this.createSceneryItem(place);
        }
      }

      this.scene.add(this.sceneryGroup);
      this.scene.add(this.territoryGroup);
      this.updateSkylineCount();
    }

    createLandmark(place) {
      const Point = place.pos;
      const tint = (place.tint[0] << 16) | (place.tint[1] << 8) | place.tint[2];
      const meshes = [];

      if (place.kind === Kinds.Business) {
        // Tower: Shaft, Roof cap, Antenna, Window band
        const shaftMat = new THREE.MeshStandardMaterial({
          color: tint,
          roughness: 0.7,
        });
        const shaft = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.0, 1.6), shaftMat);
        shaft.castShadow = true;
        shaft.receiveShadow = true;

        const roofMat = new THREE.MeshStandardMaterial({
          color: 0x5a5c62,
          roughness: 0.6,
        });
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
        // Temple/Church: Nave, Roof, Spire cone, Golden Finial
        const naveMat = new THREE.MeshStandardMaterial({
          color: tint,
          roughness: 0.8,
        });
        const nave = new THREE.Mesh(new THREE.BoxGeometry(2.0, 3.0, 1.8), naveMat);
        nave.castShadow = true;

        const roofMat = new THREE.MeshStandardMaterial({
          color: 0x6e4a41,
          roughness: 0.7,
        });
        const roof = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.4, 2.0), roofMat);
        roof.castShadow = true;

        const spireMat = new THREE.MeshStandardMaterial({
          color: 0xd6cdbe,
          roughness: 0.5,
        });
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
        // Telescope Observatory: Drum, Dome, Tilted Barrel
        const drumMat = new THREE.MeshStandardMaterial({
          color: 0xa8a8a3,
          roughness: 0.7,
        });
        const drum = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 1.6, 16), drumMat);
        drum.castShadow = true;

        const domeMat = new THREE.MeshStandardMaterial({
          color: 0xc8c8c3,
          roughness: 0.5,
        });
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
        barrel.rotation.set(0.65, 0, 0); // 55 degree tilt
      }
    }

    createSceneryItem(p) {
      const x = p.pos.x;
      const z = p.pos.z;
      const tint = (p.tint[0] << 16) | (p.tint[1] << 8) | p.tint[2];
      const theme = this.theme;

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

      const addSphere = (px, py, pz, rx, ry, rz, color) => {
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.7 });
        const m = new THREE.Mesh(new THREE.SphereGeometry(1.0, 12, 10), mat);
        m.scale.set(rx, ry, rz);
        m.position.set(px, py, pz);
        m.castShadow = true;
        this.sceneryGroup.add(m);
        return m;
      };

      if (p.kind === Kinds.House && theme === 'Suburban') {
        addBox(x, 1.4, z, 3.6, 2.8, 3.3, tint);
        addBox(x, 3.05, z, 4.1, 0.5, 3.8, 0x6e4a41);
        addBox(x, 0.65, z + 1.67, 0.65, 1.3, 0.1, 0x5b4231);
        addBox(x - 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.0, 3.4, z - 0.7, 0.45, 1.0, 0.5, 0x905542);
      } else if (p.kind === Kinds.Shop && theme === 'Suburban') {
        addBox(x, 1.1, z, 3.6, 2.2, 3.3, tint);
        addBox(x, 2.45, z, 4.1, 0.5, 3.8, 0x6e4a41);
        addBox(x, 0.65, z + 1.67, 0.65, 1.3, 0.1, 0x5b4231);
        addBox(x - 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x + 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6, 0x112233);
        addBox(x, 2.0, z + 1.9, 3.8, 0.25, 0.8, 0xe39d4e);
      } else if (p.kind === Kinds.House && theme === 'Urban') {
        const height = 4.4;
        addBox(x, height / 2.0, z, 3.6, height, 3.3, tint);
        addBox(x, height + 0.15, z, 3.8, 0.3, 3.5, 0x46484e);
        for (const band of [1.2, 2.2, 3.2]) {
          addBox(x, band, z + 1.66, 3.2, 0.4, 0.08, 0x9dcdde, 0x112233);
        }
        addBox(x + 0.9, height + 0.55, z - 0.6, 0.7, 0.5, 0.7, 0x3c3e44);
      } else if (p.kind === Kinds.Shop && theme === 'Urban') {
        addBox(x, 1.1, z, 3.6, 2.2, 3.3, tint);
        addBox(x, 2.35, z, 4.1, 0.5, 3.8, 0x5a5c62);
        addBox(x, 0.65, z + 1.67, 0.65, 1.3, 0.1, 0x46484e);
        addBox(x - 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6);
        addBox(x + 1.1, 1.55, z + 1.68, 0.7, 0.8, 0.1, 0x9dd7e6);
        addBox(x, 2.0, z + 1.9, 3.8, 0.25, 0.8, 0xc93e3a);
      } else if (p.kind === Kinds.Tree && theme === 'Suburban') {
        addBox(x, 0.85, z, 0.4, 1.7, 0.4, 0x705133);
        addSphere(x, 2.1, z, 1.05, 1.2, 1.05, 0x3b773e);
      } else if (p.kind === Kinds.Tree && theme === 'Urban') {
        addBox(x, 0.85, z, 0.15, 1.7, 0.15, 0x5a5c62);
        addBox(x, 1.85, z, 1.2, 0.8, 0.08, 0xc44a44);
      } else if (p.kind === Kinds.Bench && theme === 'Suburban') {
        addBox(x, 0.45, z, 1.4, 0.15, 0.6, 0x99693e);
        addBox(x, 0.8, z - 0.25, 1.4, 0.65, 0.12, 0x99693e);
        addBox(x - 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
        addBox(x + 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
      } else if (p.kind === Kinds.Bench && theme === 'Urban') {
        addBox(x, 0.45, z, 1.4, 0.15, 0.6, 0x503a2d);
        addBox(x, 0.8, z - 0.25, 1.4, 0.65, 0.12, 0xd6cdbe);
        addBox(x - 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
        addBox(x + 0.5, 0.2, z, 0.1, 0.4, 0.5, 0x41464a);
        addSphere(x, 0.58, z + 0.15, 0.11, 0.11, 0.11, 0xebebe8);
      } else if (p.kind === Kinds.Ball && theme === 'Suburban') {
        addSphere(x, 0.3, z, 0.3, 0.3, 0.3, 0xf17951);
      } else if (p.kind === Kinds.Ball && theme === 'Urban') {
        addBox(x, 0.3, z, 0.6, 0.6, 0.6, 0xd6b280);
        addBox(x, 0.3, z, 0.62, 0.14, 0.62, 0x4285c7);
      } else if (p.kind === Kinds.Garden && theme === 'Suburban') {
        addBox(x, 0.09, z, 0.9, 0.18, 0.9, 0x6a4d31);
        for (const offset of [-0.25, 0.0, 0.25]) {
          addSphere(x + offset, 0.22, z, 0.1, 0.175, 0.32, 0x559b3e);
        }
      } else if (p.kind === Kinds.Garden && theme === 'Urban') {
        addBox(x, 0.05, z, 0.9, 0.1, 0.9, 0x46484e);
        const heights = [0.25, 0.45, 0.7];
        for (let idx = 0; idx < heights.length; idx++) {
          const h = heights[idx];
          const off = (idx - 1.0) * 0.28;
          addBox(x + off, h / 2.0, z, 0.2, h, 0.2, 0x4285c7);
        }
      } else if (p.kind === Kinds.Flowers && theme === 'Suburban') {
        const flowerColors = [0xef799d, 0xf8cb4f, 0xa886d2];
        for (let idx = 0; idx < flowerColors.length; idx++) {
          const off = (idx - 1.0) * 0.23;
          addSphere(x + off, 0.25, z, 0.12, 0.15, 0.15, flowerColors[idx]);
        }
      } else if (p.kind === Kinds.Flowers && theme === 'Urban') {
        addBox(x - 0.15, 0.08, z, 0.4, 0.05, 0.25, 0x4c8c5a);
        addBox(x + 0.05, 0.13, z + 0.05, 0.4, 0.05, 0.25, 0x5a9e68);
        addSphere(x, 0.24, z, 0.09, 0.03, 0.09, 0xdeb85c);
      } else if (p.kind === Kinds.Mailbox && theme === 'Suburban') {
        addBox(x, 0.5, z, 0.12, 1.0, 0.12, 0x605549);
        addBox(x, 1.05, z, 0.6, 0.4, 0.45, 0x49739b);
      } else if (p.kind === Kinds.Mailbox && theme === 'Urban') {
        addBox(x, 0.5, z, 0.16, 1.0, 0.16, 0x5a5c62);
        addBox(x, 1.05, z, 0.6, 0.08, 0.45, 0xc9b27a);
      } else if (p.kind === Kinds.Pond && theme === 'Suburban') {
        addSphere(x, 0.02, z, 1.6, 0.075, 1.6, 0x58a5b8);
      } else if (p.kind === Kinds.Pond && theme === 'Urban') {
        addSphere(x, 0.02, z, 1.6, 0.06, 1.6, 0xc8c8c3);
        addSphere(x, 0.05, z, 0.9, 0.05, 0.9, 0x4285c7);
      }
    }

    initCitizens() {
      for (const y of this.yumons) {
        if (y.mesh) this.scene.remove(y.mesh);
      }
      this.yumons = [];

      for (let i = 0; i < this.citizens.length; i++) {
        const c = this.citizens[i];
        const col = c.homeParcel % 4;
        const row = Math.floor(c.homeParcel / 4);
        const px = -22.5 + col * 15.0 - 4.5 + (i % 2) * 2.5;
        const pz = -22.5 + row * 15.0 - 4.5 + Math.floor(i / 2) * 2.5;

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

        group.position.set(px, 0, pz);
        group.scale.set(0.85, 0.85, 0.85);
        this.scene.add(group);

        this.yumons.push({
          id: i,
          citizen: c,
          mesh: group,
          pos: { x: px, z: pz },
          target: { x: px, z: pz },
          behavior: Behaviors.Rest,
          targetPlace: null,
          facing: 0,
          arrived: true,
          nextDecisionTime: performance.now() + i * 2500,
          travelSeconds: 0,
          thoughtText: `${c.name} has entered the world.`,
          leftPaw,
          rightPaw,
          body,
        });
      }
    }

    initHUD() {
      // Create HTML HUD overlays on top of the 3D canvas
      this.hud = document.createElement('div');
      this.hud.className = 'universe-hud';
      this.hud.innerHTML = `
        <div class="hud-top">
          <div class="hud-badge"><span class="hud-pulse"></span><strong>Yumon Universe</strong></div>
          <button class="hud-btn" id="theme-btn" title="Toggle Suburban / Urban theme">Suburban 🌳</button>
          <div class="hud-skyline" id="hud-skyline" title="Total built skyline level">🏗️ Skyline: Lv. 0</div>
          <div class="hud-actions">
            <button class="hud-icon-btn" id="cam-reset-btn" title="Reset Camera View">🎥</button>
            <button class="hud-icon-btn" id="pause-btn" title="Pause / Play simulation">⏸️</button>
          </div>
        </div>
        <div class="hud-citizen-card" id="hud-citizen-card">
          <div class="hud-card-avatar" id="hud-card-avatar">🦊</div>
          <div class="hud-card-body">
            <div class="hud-card-header"><strong id="hud-card-name">Ember</strong> <span id="hud-card-role">Architect</span></div>
            <div class="hud-card-status" id="hud-card-status">Resting by the coffee counter</div>
          </div>
          <button class="hud-chat-btn" id="hud-chat-btn" title="Chat with this Yumon">Chat 💬</button>
        </div>
      `;
      this.container.appendChild(this.hud);

      // HUD Button Events
      const themeBtn = this.hud.querySelector('#theme-btn');
      themeBtn.addEventListener('click', () => {
        this.theme = this.theme === 'Suburban' ? 'Urban' : 'Suburban';
        themeBtn.textContent = this.theme === 'Suburban' ? 'Suburban 🌳' : 'Urban 🏢';
        this.world = generateNeighborhood(this.seed, this.theme);
        this.updateSkyAndFog();
        this.buildWorld();
        this.updateHUDCard();
      });

      const camResetBtn = this.hud.querySelector('#cam-reset-btn');
      camResetBtn.addEventListener('click', () => {
        this.camera.position.set(38, 42, 48);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
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
          chatInput.value = `Hi ${citizen.name}! What are you building today? `;
          chatInput.focus();
        }
      });

      this.updateHUDCard();
    }

    updateSkylineCount() {
      let totalLevels = 0;
      for (const lm of this.landmarks) {
        totalLevels += lm.place.activityCount || 0;
      }
      const skylineEl = this.hud?.querySelector('#hud-skyline');
      if (skylineEl) {
        skylineEl.textContent = `🏗️ Skyline: Lv. ${totalLevels}`;
      }
    }

    updateHUDCard() {
      const y = this.yumons[this.selectedCitizen];
      if (!y) return;
      const card = this.hud?.querySelector('#hud-citizen-card');
      const nameEl = this.hud?.querySelector('#hud-card-name');
      const roleEl = this.hud?.querySelector('#hud-card-role');
      const statusEl = this.hud?.querySelector('#hud-card-status');
      const avatarEl = this.hud?.querySelector('#hud-card-avatar');

      if (nameEl) nameEl.textContent = y.citizen.name;
      if (roleEl) roleEl.textContent = y.citizen.role;
      if (statusEl) {
        const actionLabel = BehaviorLabels[this.theme][y.behavior] || y.behavior;
        const targetNoun = y.targetPlace !== null ? Nouns[this.theme][this.world.places[y.targetPlace].kind] : 'neighborhood';
        statusEl.textContent = `${actionLabel} at the ${targetNoun} · ${y.citizen.savings} coins`;
      }
      if (avatarEl) {
        const icons = ['🦊', '🐬', '🌿', '✨'];
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
          this.spawnSparkles(y.pos.x, 1.2, y.pos.z, y.citizen.color);
          return;
        }
      }

      // Check Landmarks
      for (const lm of this.landmarks) {
        const intersects = this.raycaster.intersectObjects(lm.meshes, true);
        if (intersects.length > 0) {
          // Direct a nearby Yumon to this landmark!
          const y = this.yumons[this.selectedCitizen];
          y.targetPlace = lm.place.id;
          y.behavior = Behaviors.Build;
          y.target = approachPlace(y.pos, lm.place);
          y.arrived = false;
          y.travelSeconds = 0;
          this.updateHUDCard();
          this.spawnSparkles(lm.pos.x, 2.0, lm.pos.z, 0xffd166);
          return;
        }
      }
    }

    spawnSparkles(x, y, z, colorHex) {
      for (let i = 0; i < 8; i++) {
        const pGeo = new THREE.SphereGeometry(0.08, 6, 6);
        const pMat = new THREE.MeshBasicMaterial({ color: colorHex });
        const p = new THREE.Mesh(pGeo, pMat);
        p.position.set(
          x + (Math.random() - 0.5) * 0.8,
          y + (Math.random() - 0.5) * 0.8,
          z + (Math.random() - 0.5) * 0.8
        );
        p.userData = {
          vy: 0.8 + Math.random() * 1.5,
          vx: (Math.random() - 0.5) * 1.2,
          vz: (Math.random() - 0.5) * 1.2,
          life: 1.0,
        };
        this.sparklesGroup.add(p);
      }
    }

    tickSimulation(dt, now) {
      if (this.paused) return;

      // Update sparkles
      for (let i = this.sparklesGroup.children.length - 1; i >= 0; i--) {
        const p = this.sparklesGroup.children[i];
        p.userData.life -= dt * 1.5;
        p.position.x += p.userData.vx * dt;
        p.position.y += p.userData.vy * dt;
        p.position.z += p.userData.vz * dt;
        p.scale.multiplyScalar(0.96);
        if (p.userData.life <= 0) {
          this.sparklesGroup.remove(p);
        }
      }

      // Tick citizens
      for (let i = 0; i < this.yumons.length; i++) {
        const y = this.yumons[i];

        // Autonomous decision making
        if (y.arrived && now >= y.nextDecisionTime) {
          this.decideNextAction(y);
        }

        // Movement towards target
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
          const proposed = {
            x: y.pos.x + dirX * step,
            z: y.pos.z + dirZ * step,
          };

          if (isWalkable(proposed, this.world.places)) {
            y.pos = proposed;
          } else {
            // Slide along axes
            const xOnly = { x: proposed.x, z: y.pos.z };
            const zOnly = { x: y.pos.x, z: proposed.z };
            if (Math.abs(dirX) > 0.01 && isWalkable(xOnly, this.world.places)) {
              y.pos = xOnly;
            } else if (Math.abs(dirZ) > 0.01 && isWalkable(zOnly, this.world.places)) {
              y.pos = zOnly;
            } else {
              y.target = y.pos;
              y.arrived = true;
              y.nextDecisionTime = now + 1500;
            }
          }

          // Walk animation (bobbing + feet walking)
          const walkCycle = Math.sin(now * 0.01 + y.id * 2.0);
          y.mesh.position.set(y.pos.x, 0.45 + Math.abs(walkCycle) * 0.16, y.pos.z);
          y.mesh.rotation.y = y.facing;
          y.mesh.rotation.z = Math.sin(now * 0.008 + y.id) * 0.08;
          if (y.leftPaw) y.leftPaw.position.z = walkCycle * 0.12;
          if (y.rightPaw) y.rightPaw.position.z = -walkCycle * 0.12;
        } else if (!y.arrived) {
          y.arrived = true;
          this.onYumonArrived(y);
        } else {
          // Idle / Active animation
          if (y.behavior === Behaviors.Build) {
            // Jump and hammer sparkles!
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

    onYumonArrived(y) {
      if (y.targetPlace !== null) {
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
      // Find nearby interesting places
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
        y.target = approachPlace(y.pos, choice.place);
        y.arrived = false;
        y.travelSeconds = 0;
      } else {
        // Explore neighborhood
        y.behavior = Behaviors.Explore;
        y.targetPlace = null;
        const angle = Math.random() * Math.PI * 2.0;
        const dist = 4.0 + Math.random() * 6.0;
        const target = {
          x: Math.max(-28, Math.min(28, y.pos.x + Math.cos(angle) * dist)),
          z: Math.max(-28, Math.min(28, y.pos.z + Math.sin(angle) * dist)),
        };
        if (isWalkable(target, this.world.places)) {
          y.target = target;
          y.arrived = false;
          y.travelSeconds = 0;
        }
      }

      if (y.id === this.selectedCitizen) {
        this.updateHUDCard();
      }
    }

    // Called when RSS Chat or Products update
    onRssStateUpdate(state) {
      if (state.chat_busy) {
        // Yumon shows thinking action
        const activeYumon = this.yumons[this.selectedCitizen];
        if (activeYumon) {
          activeYumon.behavior = Behaviors.Inspect;
          this.spawnSparkles(activeYumon.pos.x, 1.2, activeYumon.pos.z, 0x9dd8b3);
          if (this.hud?.querySelector('#hud-card-status')) {
            this.hud.querySelector('#hud-card-status').textContent = `Thinking about your idea... 💭`;
          }
        }
      } else if (state.chat && state.chat.length > 0) {
        const lastMsg = state.chat[state.chat.length - 1];
        if (lastMsg.role === 'yumon') {
          // Celebrate response!
          const activeYumon = this.yumons[this.selectedCitizen];
          if (activeYumon) {
            this.spawnSparkles(activeYumon.pos.x, 1.5, activeYumon.pos.z, 0xf8cb4f);
          }
        }
      }
    }

    animate() {
      requestAnimationFrame(this.animate);
      const now = performance.now();
      const dt = Math.min(0.1, (now - (this.lastTime || now)) / 1000.0);
      this.lastTime = now;

      this.tickSimulation(dt, now);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    }
  }

  // Initialize on DOM load
  window.addEventListener('DOMContentLoaded', () => {
    const universeContainer = document.getElementById('universe-container');
    if (universeContainer) {
      window.__yumon_universe_app = new YumonUniverseApp(universeContainer);
    }
  });

  // Global callback for state sync
  window.__onUniverseRssSync = function (state) {
    if (window.__yumon_universe_app) {
      window.__yumon_universe_app.onRssStateUpdate(state);
    }
  };
})();
