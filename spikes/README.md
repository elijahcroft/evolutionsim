# Spikes

Throwaway experiments. **Nothing here is imported by `src/evosim/`, and nothing here should be.**
A spike exists to answer a design question that is hard to answer on paper, and then to be
deleted or rewritten when the real thing is built. Spikes are allowed to be ugly, untested, and
dependency-free.

---

## `morphology/` — procedural 3D creature bodies

**Question it exists to answer:** ej wants evolvable 3D body plans eventually. Before committing
to a morphology genome in a later milestone, three things needed checking:

1. Which morphology genes actually earn their place?
2. Do *small* mutations produce plausible, varied bodies — is morphology space smooth enough for
   gradual evolution, or does it fall off a cliff into garbage?
3. Can a lineage be walked from a simple blob to something complex by selection alone?

**How to run it:** open `morphology/index.html` in a browser. No build step, no dependencies —
raw WebGL2 with hand-rolled matrix maths and mesh generation. (Note: the Chrome automation
extension cannot open `file://` URLs, so testing it from a tool requires
`python3 -m http.server` in that directory. Opening the file directly works fine by hand.)

The large creature is the parent; the six below are its mutant offspring. Click an offspring to
promote it — that walks a lineage under your selection. Sliders edit the parent directly. The
founder is deliberately a plain green blob with no limbs, head, or segments, so everything you
see appear is mutation plus selection.

### Findings

**It works, and the space is smooth at low mutation strength.** Seven generations of hand
selection took the founder blob to an elongated, tailed body with a head, an eye, a dorsal fin
and a limb. Offspring visibly resemble their parent — heritability reads correctly at a glance,
which is the property that matters. 19 genes, ~4k triangles per creature, no console errors.

**It degenerates at high mutation strength, in an instructive way.** At 3x strength, genes
saturate against their bounds within a few generations (`spine_arch` pinned at its maximum,
`body_radius` collapsed toward its minimum) and bodies become flat crescent slivers that read as
nothing biological. Two lessons for the real implementation:

- Per-locus sigma must stay small relative to locus span. This is the same constraint
  `config.py` already enforces for the existing genome (sigma <= span/4), and it is clearly the
  right guard.
- **Gene bounds that are individually reasonable can be jointly absurd.** `spine_arch = 1.0` is
  fine on a thick body and grotesque on a thin one. Morphology genes will need either tighter
  bounds or explicit joint constraints — an issue the current scalar-only genome does not have,
  and a real cost of adding morphology that I had not anticipated.

**Which genes earned their place:** `body_length`, `body_radius`, `fullness`, `taper`,
`segment_count`/`segment_depth`, `radial_symmetry`, `limb_pairs`, `limb_length`, `limb_splay`,
`head_size`, `tail_length`, `dorsal_fin`. These each change the silhouette recognisably.

**Which did not pull their weight:** `limb_droop` and `limb_segments` are nearly invisible at
typical limb lengths, and `eye_size` is decorative — it changes nothing about the animal's
read at body scale. If morphology lands in the simulation, these three are candidates to cut,
since every extra locus is genome size, mutation surface, and UI clutter.

**Cost estimate for the real transition:** the genotype-to-geometry function
(`buildCreature`, ~130 lines) is the transferable part. The expensive parts are *not* the mesh:
they are (a) making morphology enter the energy equations so it is not decoration — limb count
into locomotion cost, surface-area-to-volume into thermoregulation, and (b) the joint-constraint
problem above. Confirms the original estimate that this belongs after M5, not before.

### Deliberately absent

No energy costs, no fitness, no simulation. Selection here is your mouse. In the real thing,
morphology must feed the cost equations or it is just decoration — which is exactly the trap
this spike is meant to keep us out of.
