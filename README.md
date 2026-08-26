# crowdbake

Vertex Animation Textures (VAT) for large animated crowds in the browser, plus the
bone-matrix variant, plus the bits that usually get left out of a VAT demo:
LODs that share one animation texture, chunked frustum culling, shadows that
match the animation, GPU cross-fade between clips, socket attachments, and
stride-matched playback so the feet do not slide.

Everything runs with no external assets — the pipeline ships a procedural rigged
humanoid so `npm install && npm run dev` gives you a crowd immediately. It also
bakes real rigged glTF/GLB.

```
npm install
npm run dev        # bakes the demo asset, then serves the demo
```

Other entry points:

| command | what it does |
| --- | --- |
| `npm run bake -- <file.glb> --out public/baked/x` | bake your own rigged glTF |
| `npm run export:glb` | write the procedural rig out as a real animated `.glb` |
| `npm run verify` | decode the bake and compare against CPU ground truth |
| `npm run inspect` | human-readable dump of a bake (clips, strides, LODs, memory) |
| `npm test` | 24 offline checks: numerics, stride, LODs, container, glTF round-trip |
| `npm run test:gpu` | headless-browser smoke test of every runtime path + screenshots |
| `npm run bench` | CPU cost per frame at 1k…100k instances, no renderer |
| `npm run bake:goobers` | capture + bake the SDF critters from MGameExample/goobers |

---

## The technique

**Bake.** Sample every clip at a fixed rate. At each frame, evaluate skinning on
the CPU and write the resulting *vertex positions* into a texture: one texel per
vertex per frame. A second texture holds normals. At runtime nothing is skinned —
the vertex shader just reads two frames and interpolates.

```
texel index = frameIndex * vertexCount + vertexId
```

which is linearised into a 2D texture, so vertex count and frame count can each
exceed the 16384 texture limit independently.

**Runtime.** One `InstancedBufferGeometry` per chunk. Per instance: one
interleaved 16-float record.

```
0..3    world x, y, z, yaw
4..7    clipA, phaseOffsetA, rateA, scale
8..11   clipB, phaseOffsetB, rateB, fadeStartTime
12..15  shirt rgb, packed variant (pants | skin | accessory)
```

The vertex shader resolves clip + phase into two frame rows, `texelFetch`es both,
mixes, decodes from the bounding box, then rotates/scales/translates by the
instance. Cross-fade runs the same path a second time against clip B and mixes.

Phase is **not** uploaded. It is computed on the GPU as `fract(offset + time * rate)`,
so a crowd that is standing still uploads nothing at all and keeps animating.

Shadows are the same displacement duplicated into a `MeshDepthMaterial` assigned
as `customDepthMaterial`, so the shadow matches the animation exactly.

### Two playback modes, same asset

Bake with `--bones` and you get both; switch at runtime with `crowd.reconfigure({ mode })`.

| | **VAT** (`mode: 'vat'`) | **bone matrix** (`mode: 'bone'`) |
| --- | --- | --- |
| stores | vertex positions + normals per frame | 3 texels per bone per frame |
| demo asset size | **7.80 MB** | **338 KB** (24× smaller) |
| scales with | vertex count | bone count |
| vertex shader | 4 texel fetches | ~24 (4 influences × 2 frames × 3 rows) |
| attachments | no | yes |
| best for | dense crowds | anything needing sockets, or huge meshes |

They compose: the demo runs **VAT bodies with bone-driven attachments**, because
the bone texture is cheap enough to keep around purely for sockets. The caps and
balloons in the demo are instanced over the same per-chunk buffers as the bodies,
so they cull, LOD and cross-fade for free; instances not wearing an accessory
collapse to a degenerate triangle.

---

## The parts that actually bite

**fp16 precision.** Naive world-space half floats look broken. Positions are
normalised to the bake's bounding box and decoded with a uniform. `npm run verify`
measures the result end to end: max **0.739 mm** error over 681,696 vertex-frames,
against an fp16 quantum of 1.068 mm.

**`texelFetch`, not filtered sampling.** Frames are lerped by hand. No
`OES_texture_float_linear`, no half-texel offsets to get wrong.

**Texture size cap.** The linearised index means a bake only fails when
`vertexCount × totalFrames` exceeds 16384², and the baker widens the texture
automatically before it gets there (tested down to `--tex-width 64`).

**Culling.** An `InstancedMesh` has one bounding sphere, so walking into a 100k
crowd draws all 100k. Instances are counting-sorted into a spatial grid each
frame; each chunk gets its own sphere, its own draw and its own LOD. Set
`chunkSize >= worldSize` to collapse to the classic single draw call — the demo
exposes this as a slider so you can watch the trade-off. (In single-chunk mode
LOD selection also collapses, since one sphere spans the whole world.)

**Where the LOD distance is measured from.** Chunk-granular LOD has a trap: if
you measure to the chunk's *near edge* — the conservative choice, so no instance
is ever under-detailed — then a 42 m chunk holds LOD0 until its centre is 52 m
away, and essentially the whole crowd renders at full detail. Measuring from the
centre (`lodPivot: 'centre'`, the default) and using smaller chunks for
granularity is what you actually want. This one setting was an 11x difference in
submitted triangles; see *Performance* below.

**Stride matching.** The interesting one. For each clip the baker finds the
contact feet, takes the median backwards velocity across every planted frame, and
multiplies by the clip duration to get metres travelled per cycle. Playback rate
is then `speed / stride`. Using the median rather than integrating means running
clips with an airborne phase come out right too.

On the procedural rig — where the gait is built from an explicit foot trajectory
solved with 3D two-bone IK, so the true stride is known analytically — the
extractor recovers it exactly:

```
walk   baked 1.4800 m   truth 1.4800 m   -0.00%
jog    baked 2.2800 m   truth 2.2800 m   -0.00%
run    baked 3.3000 m   truth 3.3000 m   -0.00%
```

Toggle **stride matching** off in the demo to see the same crowd play its clips at
the authored rate and skate across the ground.

---

## Measured

Machine: AMD Ryzen 7 7735HS, Node 24, Windows 11.

**Bake** (`npm run bake:demo`) — 2,367 verts, 4,592 tris, 25 bones, 6 clips × 48
frames = 288 frames, in **175 ms**:

```
texture       4096 x 167   (681,696 texels)
  positions   5.20 MB  RGBA16F
  normals     2.60 MB  RG16F           <- octahedral, 2 channels not 3
  bones       337.5 KB RGBA32F
  per vertex  12.0 bytes/vertex/frame
lods          4592 / 1750 / 484 / 140 / 44 triangles   (one shared texture)
```

LODs are built by vertex clustering keyed on cell + dominant bone + material, so
every surviving vertex keeps its original id and therefore its texel column. An
LOD costs an index buffer and a few static attributes — **zero** extra animation
memory.

**Runtime CPU cost** (`npm run bench`) — medians of 100 frames, no renderer
involved, 24 m chunks, agents wandering with steering and obstacle avoidance:

| instances | sim | of which separation | chunk + LOD | total | best frame | upload |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.21 | 0.12 | 0.19 | **0.42 ms** | 0.27 | 0.06 MB |
| 10,000 | 1.62 | 0.89 | 0.51 | **2.14 ms** | 1.92 | 0.61 MB |
| 25,000 | 5.36 | 3.50 | 1.58 | **6.93 ms** | 5.99 | 1.53 MB |
| 50,000 | 3.54 | off | 3.27 | **6.95 ms** | 5.56 | 3.05 MB |
| 100,000 | 7.51 | off | 9.93 | **17.58 ms** | 13.27 | 6.10 MB |

Neighbour separation is the most expensive behaviour and auto-disables above
40,000 agents; it is a toggle in the demo panel.

**Submitted geometry** at 20,000 agents, before frustum culling:

| configuration | triangles |
| --- | ---: |
| 3 LODs bottoming out at 349 tris, near-edge pivot, 42 m chunks | 19.8 M |
| 5 LODs bottoming out at 44 tris, centre pivot, 24 m chunks | **1.78 M** |

`npm run bench` prints the LOD split alongside the timings, which is the quickest
way to see whether your `lodDistances` are doing anything:

```
  instances    triangles     L0 (4592t)  L1 (1750t)   L2 (484t)   L3 (140t)    L4 (44t)
      10000      0.91 M              0          74         369        1847        7710
     100000      9.12 M              0         793        3630       18391       77186
```

**Honest caveat about frame rates.** This machine has no usable discrete GPU, so
the only renderer available was SwiftShader (a software rasteriser). Absolute fps
here means nothing for your hardware. What it *can* do is measure the same scene
before and after a change, and at 4,000 agents that came out at **1.83 fps before
the optimisation pass and 9.73 fps after** — a 5.3x improvement that is almost
entirely geometry the GPU no longer has to touch. Press `B` in the demo to get
real numbers on your own GPU.

---

## Hosting

`index.html` in the repo root is a **Vite source file, not a hostable page** — it
points at `/src/main.js` and uses bare imports (`three`, `lil-gui`) that no
browser can resolve, and the baked asset does not exist until the build runs it.

Build once, then deploy `dist/`:

```
npm install
npm run build      # bakes the asset, then bundles -> dist/
```

That is the whole deployable, four files:

```
dist/index.html             1.5 KB
dist/assets/index-*.js       592 KB   (149 KB gzipped)
dist/baked/crowd.json        8.7 KB
dist/baked/crowd.bin        8.40 MB   (4.65 MB gzip, 3.50 MB brotli)
```

It is a plain static site — no server, no build step at request time, no
COOP/COEP headers, no special MIME types. Drop `dist/` on GitHub Pages, Netlify,
Cloudflare Pages, S3, nginx, anything.

**Two things worth knowing.**

*Enable compression for `.bin`.* It is 8.4 MB raw and 3.5 MB brotli — a 58%
saving on by far the largest file. Most hosts do this automatically; nginx and S3
need `.bin` added to the compressible types list.

*Subdirectory hosting needs the trailing slash.* The default build uses relative
paths, so `https://example.com/demo/` works but `https://example.com/demo` does
not — the browser resolves `./assets/index-*.js` against `/` and 404s. Almost
every host 301-redirects the missing slash for you, so this is usually invisible.
If yours does not, build with an absolute base instead, which works either way:

```
npm run build -- --base=/demo/
```

(Verified both ways against a deliberately dumb static server that does no
redirecting: relative build serves correctly from the root and from `/demo/`;
absolute-base build additionally survives `/demo`.)

**Requirements:** WebGL2, which means any browser since ~2017. The demo needs no
network access after load and stores nothing.

---

## Bake CLI

```
node tools/bake.mjs [input.gltf|input.glb] [options]

  --procgen           bake the built-in procedural humanoid (default with no input)
  --out <prefix>      writes <prefix>.json and <prefix>.bin
  --fps <n>           sampling rate per clip (default 30)
  --frames <n>        fixed frame count per clip, overrides --fps
  --clips a,b,c       only bake these clips
  --lods <n>          number of LOD levels (default 3)
  --lod-ratios a,b,c  explicit vertex ratios, e.g. 1,0.4,0.15,0.05
  --bones             also bake a bone-matrix texture (attachments + tiny memory)
  --tangents          bake per-frame tangents for normal mapping
  --tex-width <n>     animation texture width (default 4096)
  --density <f>       procedural mesh density (default 1.4)
  --skin <n>          which glTF skin to use (default 0)
```

The glTF reader is dependency-free and handles `.gltf` + `.glb`, external and
embedded buffers, byte-strided and normalised accessors, sparse accessors, and
LINEAR / STEP / CUBICSPLINE animation samplers. It does **not** do Draco or morph
targets.

---

## Runtime API

```js
const asset = await VATAsset.load('baked/crowd.json');

const crowd = new VATCrowd(asset, {
  capacity: 100000,
  mode: 'vat',              // or 'bone'
  chunkSize: 24,            // >= worldSize for one draw call
  worldSize: 420,
  lodDistances: [8, 20, 45, 100],
  lodPivot: 'centre',       // 'edge' is conservative and very expensive
  shadowMaxLod: 2,          // chunks past this LOD stop casting
  farPool: true,            // last-LOD chunks merge into one unculled draw
  crossFade: true,
  crossFadeMaxLod: 1,       // far LODs skip the extra fetches entirely
  fadeDuration: 0.28,
  castShadow: true,
});
scene.add(crowd);

crowd.count = 20000;
crowd.setTransform(i, x, y, z, yaw);
crowd.setScale(i, 1.05);
crowd.setVariant(i, { shirt: [r, g, b], pants: 3, skin: 1, accessory: 1 });
crowd.play(i, clipIndex, { rate });          // cross-fades
crowd.setSpeed(i, metresPerSecond);          // stride-matched, phase-continuous
crowd.update(dt, camera);
```

Socket attachments (needs a `--bones` bake):

```js
crowd.addAttachment(new VATAttachment(capGeometry, {
  socket: 'head',                            // name from the manifest, or a bone index
  offset: new Matrix4().makeTranslation(0, 1.775, 0.004),
  accessory: 1,                              // which packed variant wears it
  maxLod: 1,
}));
```

`crowd.stats` reports simulated / submitted / **rasterised** instance counts
separately, so you can see what culling actually saved. (`renderedInstances` is
accumulated in `onBeforeRender`, which is the only place three exposes the
post-frustum truth.)

---

## The paint minigun

Press <kbd>G</kbd> (or pick **gun** under camera). Click to lock the pointer,
hold LMB to fire, WASD to move, <kbd>E</kbd>/<kbd>Q</kbd> to fly up and down —
gun mode keeps whatever altitude you enter it at, so strafing runs from the air
work. <kbd>R</kbd> respawns the crowd to the editor count. Infinite ammo, no
spread: every round is an instant laser flash in its own rainbow hue, straight
to the crosshair, and whatever it hits erupts in a fountain of paint that
speckles the floor for good.

How it works, since each piece is a pattern worth stealing:

- **Hit detection** is analytic, not mesh raycasting. Every instance is a
  cylinder derived from its interleaved record (position, yaw, scale x the
  asset's baked bounds), and the crowd's chunk grid -- rebuilt every frame for
  culling anyway -- doubles as the broadphase. A round samples grid cells along
  its XZ track and tests a few dozen candidates, not 100,000.
- **Kills** swap the last live agent into the dead slot across both the sim
  arrays and the 64-byte instance record, so indices stay dense and nothing
  else notices. Slots beyond the count keep their state, which is why
  <kbd>R</kbd> resurrects the fallen exactly where they fell.
- **Tracers and droplets** are fire-and-forget GPU ring buffers: the CPU writes
  one record at spawn and the vertex shader evaluates ballistic flight from
  `uTime` alone. A screenful of paint spray costs no per-frame CPU.
- **Floor paint** accumulates in one world-covering render target; a splat is
  "draw a blob quad into the target once", so ten thousand splats render at the
  same cost as one. Droplet landing times are solved analytically at spawn and
  scheduled as speckles, so the ring appears exactly where the droplets land.

---

## Shipping as one file

`npm run build:single` makes **`dist/index.html` itself the whole app** in a
single HTML file (bundle inlined, bakes embedded as base64) -- post that one
file anywhere and it runs; `dist/crowdbake.html` is the same bytes under a
name that survives being dropped next to an existing site. ~39 MB with the
humanoid and the three shipped goobers; add `-- --assets crowd` for a lean
~11 MB file. It boots from a plain `file://` open, so it works anywhere that
serves bytes. The menagerie stays multi-file (40 kinds is a quarter gigabyte);
keeping `dist/baked/` next to the single file re-enables `?asset=menagerie`.

---

## Baking things that are not skinned rigs (goobers)

`tools/bake-goober.mjs` bakes the SDF blend-shell critters from
`C:\Projects\MGameExample\goobers` — characters with **no bones and no glTF**,
whose geometry is a vertex shader snapping a template mesh onto a smooth-min SDF
and whose animation is entirely procedural. This is the general recipe for
baking any procedurally animated thing:

1. The goobers app is loaded headless with `requestAnimationFrame` stubbed out,
   so it initialises but never runs. `critter.update()` is driven at a fixed
   timestep with a seeded RNG (same seed, same goober), steering the critter in
   a straight line, snapshotting its prim uniforms each frame.
2. The best loop inside the capture is found by comparing prim signatures
   (pose + velocity, so the seam does not pop), and the walk clip's stride is
   the distance the critter *actually travelled* over that loop — stride
   matching then works exactly as it does for the humanoid.
3. The SDF vertex snap is ported to plain JS (`tools/lib/goober-sdf.mjs`) and
   evaluated per frame in Node. Eyes are rigid spheres placed directly — they
   are not part of the SDF in the original either, and blending them in makes
   the shell's tuck logic carve craters around the embedded pupil.
4. The result is a frames dump that `bake.mjs --frames-dump` bakes like any
   rig: same fp16 texture, same LOD chain (the prim id stands in for the
   dominant bone so limbs do not weld), same runtime. No bone texture, so no
   attachments and no bone playback mode — the demo greys those out.

```
npm run bake:goobers      # the three shipped kinds (biped, quad, cat)
npm run bake:menagerie    # every kind (~40), plus the menagerie manifest
```

Then pick the asset in the demo panel, or `?asset=goober-biped`. Vertex colors
carry the goober's identity; instances get a subtle per-instance hue shift
instead of the humanoid's shirt/pants palette.

**The menagerie** (`?asset=menagerie`) instances every baked kind together in
one scene: one small VATCrowd + sim per kind, sharing the world. The instances
slider is the total across all kinds, and each kind's gait ladder is scaled by
its own captured walk speed, so cats trot and golems trudge. Known limitation:
separation only acts within a kind, so different kinds can overlap in dense
clusters.

**Live regeneration (dev server).** `npm run dev` adds a `goober lab` panel:
pick any of the ~40 kinds, *bake this kind* (fixed seed) or *regenerate (new
look)* (random seed - same species, new body). The dev server captures and
bakes on demand (15-60 s) and reloads when done; regenerating from inside the
menagerie drops the new look straight back into it. The goobers app is vendored
at `vendor/goobers/`, so all of this works from a fresh clone.

---

## Testing

The interesting problem with a technique like this is that most of it cannot be
checked by looking at it. Three layers:

**`npm test` — 24 offline checks.** Half-float round-trip against the IEEE 2⁻¹¹
bound plus subnormal/overflow behaviour; octahedral encoding through fp16;
the procedural gait's planted foot (asserts the contact foot's lateral drift stays
under 0.1 mm/frame and its backwards velocity is constant to 5e-3); stride
extraction against analytic ground truth; LOD topology and monotonicity; container
alignment; clip frame ranges tiling the texture.

The load-bearing one: the procedural rig is exported to a real `.glb` and re-baked
through the glTF reader, then compared to the direct bake texel by texel. They
agree on 99.99% of 681,696 texels, and the remainder differ by at most one fp16
ULP — which is what makes the glTF path trustworthy rather than merely written.

**`npm run verify` — numeric round-trip.** Decodes the baked texture exactly as
the vertex shader will and compares against freshly CPU-skinned ground truth. This
is what catches bounding-box and fp16 mistakes without a GPU.

**`npm run test:gpu` — headless GPU smoke test.** Boots the built app in headless
Chrome, fails on any console error or GL warning, and exercises every runtime
toggle: single-chunk vs chunked, VAT vs bone mode, cross-fade on/off, LOD bias,
stride matching, all five crowd scenarios, and walking inside the crowd. It
asserts the crowd is genuinely in the shadow pass (by differencing draw calls with
shadows on and off — note three resets `renderer.info` *after* the shadow pass, so
the default counters silently hide every shadow draw), and that the resulting
frame is not blank. Screenshots land in `tools/smoke-out/`.

---

## Performance notes

Written down because they cost real time to find.

**The frame counter was lying.** The demo clamps its simulation `dt` so a long
frame cannot teleport the crowd, and the HUD was computing fps from that same
clamped value — so it could never report below `1/clamp` and read exactly
"20.0 fps" no matter how slow the browser actually was. The HUD now takes wall
clock time for reporting and the clamped value only for the sim, and it shows the
JS cost next to the true frame time so you can see at a glance whether you are
CPU- or GPU-bound.

**LOD pivot and chain depth dominated everything else.** Two compounding
mistakes: LOD measured from the chunk's near edge (so a 42 m chunk stayed at LOD0
until its centre was 52 m away) and a chain whose cheapest level was still 349
triangles. 10,308 far-away agents were costing 3.6 M triangles between them.
Fixing both took submitted geometry from 19.8 M to 1.78 M triangles.

**Then it stopped being geometry-bound.** After the LOD fix, halving triangles
again via `lodBias` changed nothing measurable, but turning shadows off gained
32%. The shadow pass was an exact duplicate of the colour pass's draw calls, so
`shadowMaxLod` (default 2) stops distant chunks casting — a character 45 m away
is a few pixels in the shadow map but still costs a whole extra draw. Worth 21%.

**"Models disappear when I zoom out" was the fog.** Frustum culling was
suspected and formally cleared — an independent CPU frustum test of every chunk
sphere matched three's post-cull count exactly at every camera distance. The
culprit was `FogExp2(density 0.0042)`, tuned at eye level: a character 250 m away
was 67% blended into the fog colour, 91% at 370 m. Density is now 0.0011 (the far
corner of the world fogs ~1/3) and a GUI slider. Moral: before debugging culling,
render the frame and look at it.

**The far pool: one draw for everything at the last LOD.** Once LODs work, a
zoomed-out camera sees ~290 chunks x 2 passes of a 44-triangle mesh — pure
draw-call overhead. Chunks classified at the last LOD now collapse into a single
unculled instanced draw: **288 draws -> 7** at full zoom-out. The pool only
rebuilds on frames where the camera or an instance moved, and LOD transitions
can only happen on those frames, so it never disagrees with per-chunk
visibility. Caveat measured honestly: under SwiftShader the pool is ~10% *slower*
(a software rasteriser has no draw-call overhead but pays full price for the
pool's unculled vertex work); on real WebGL drivers, where each draw costs
10-50 us of CPU, the economics invert. It is a GUI toggle — A/B it on your GPU.

**Per-cell dirty tracking.** "Did anything move" used to be a global flag, so one
moving agent re-gathered and re-uploaded every chunk. Movement is now tracked per
instance and consumed per cell: cells where nothing moved skip their gather and
upload entirely, and the sim skips writing agents below 1 mm/s. A settled crowd
uploads nothing and keeps animating (phase lives on the GPU).

**Stale cross-fades rendered the wrong clip at distance.** An instance that
finished a fade kept the old clip in slot A forever — invisible in cross-fade
materials (blend clamps to 1) but LODs past `crossFadeMaxLod` compile without
cross-fade and render slot A: a walker that faded to idle long ago would walk
again the moment its chunk dropped to LOD 2. A FIFO of active fades now folds
each fade into slot A the frame it completes.

**Smaller chunks beat fewer draw calls.** Sweeping chunk size: 24 m gave 131 draws
and 9.73 fps, 64 m gave 29 draws and 8.40 fps. Fewer draws lost to worse frustum
culling, because a big chunk drags a lot of off-screen instances on screen with
it. The reported triangle count is pre-cull, which hid this at first.

**The instance store had a cache cliff.** Per-instance state was four separate
arrays; building a chunk's buffer is a gather, so that was four cache misses per
instance and cost went super-linear once the working set outgrew L3 — 100k agents
cost 202 ms/frame. One interleaved 64-byte record made it 18.8 ms.

**Beware benchmarking next to a software rasteriser.** A leaked headless browser
kept rasterising in the background and moved Node benchmark means by 3-4x. The
benchmark now reports medians and the smoke test always tears its browser down.

---

## Limits

- No runtime IK, no retargeting, no procedural look-at. VAT frames are baked.
- Attachments require the bone texture. Pure VAT has no sockets.
- Cross-fade costs 2× the fetches while it is running; `crossFadeMaxLod` keeps
  distant LODs on the cheap path.
- Culling is per-chunk on the CPU. WebGL2 has no indirect draw, so there is no
  GPU culling path here; that needs WebGPU.
- Single-chunk mode disables LOD selection as a side effect (one sphere, one
  distance).
- The baker does not read Draco-compressed or morph-target glTF.
- At 100k the demo is CPU-bound on the sim and the instance gather (~18 ms/frame)
  before the GPU does anything. 50k is comfortable; 100k needs separation off.
- LOD is per chunk, so one chunk is one detail level. Smaller chunks buy finer
  granularity and better culling at the cost of more draw calls.
- No imposter/billboard tier below the last mesh LOD. The cheapest level is 44
  triangles; a camera-facing quad would be cheaper still for very distant crowds.
- The far pool is not frustum-culled (it spans the world by construction), so
  walking inside a huge crowd pays vertex cost for far instances behind the
  camera — 44 tris each, cheap on hardware, but it is the trade being made.

---

## Layout

```
tools/
  bake.mjs            VAT baker CLI
  verify.mjs          decode-and-compare against CPU ground truth
  inspect.mjs         readable dump of a bake
  export-gltf.mjs     procedural rig -> real animated .glb
  test.mjs            offline test suite
  smoke.mjs           headless-browser GPU smoke test
  bench-cpu.mjs       renderer-free CPU benchmark
  lib/
    procgen.mjs       procedural rigged humanoid + IK gaits
    gltf.mjs          dependency-free glTF/GLB reader
    rig-gltf.mjs      glTF -> RigModel with clip sampling
    skin.mjs          CPU linear-blend skinning + tangents
    stride.mjs        contact-foot stride extraction
    decimate.mjs      LOD by vertex clustering (preserves vertex ids)
    half.mjs          fp32 -> fp16
    octa.mjs          octahedral normal encoding
    pack.mjs          binary container writer

src/vat/
  VATAsset.js         manifest + binary -> textures and LOD templates
  VATCrowd.js         chunking, LOD, instance store, upload
  VATAttachment.js    bone-socket attachments
  vat-shader.js       GLSL injection for both modes + depth material

src/demo/
  crowd-sim.js        steering, separation, speed -> clip -> stride-matched rate
  env.js              sky, ground, sun with a texel-snapped following shadow map
  camera-rig.js       orbit / walk / tour
  hud.js, benchmark.js, props.js
```
