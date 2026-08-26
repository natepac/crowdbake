# goobers (vendored)

`goobers.html` is the self-contained "SDF blend-shell" character toy from
`MGameExample/goobers`, vendored here so `tools/bake-goober.mjs` can capture and
bake its critters without an external checkout. Single file, no build step; its
only dependency (three.js from a CDN importmap) is intercepted and served from
this repo's node_modules during baking.

List bakeable kinds:   node tools/bake-goober.mjs --list
Bake one:              node tools/bake-goober.mjs --kind cat --seed 4 --out tools/_out/cat
