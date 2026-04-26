# Dummy Kinematics

A Blender 5.0+ add-on that gives you toggleable, IK-feeling limb posing
without a real IK solver. You hand-pose each limb in seven reference
directions (Forward, Back, Left, Right, Up, Down, Retract) once. After
that, save a target world-space position for the limb's end bone at one
frame, move the rig at another frame, and Restore — the addon blends
your captured poses to land the end bone at (or as close as possible
to) the saved target.

Because every input pose was hand-authored by you, the output always
looks natural — knees bend the right way, no IK flipping, no pole
targets, no chain configuration.

## Install

**Via the Blender Extensions Platform:**

1. `Edit → Preferences → Get Extensions`
2. Search for "Dummy Kinematics"
3. Click Install

**Manually from a release zip:**

1. Download the latest zip from this repo's releases.
2. `Edit → Preferences → Add-ons → Install from Disk`, pick the zip.

## Quick start

1. Select your armature, enter **Pose Mode**.
2. Open `View3D → Sidebar → Dummy Kin`.
3. **Add Limb**, pick the end bone you want to plant (foot, hand, tail
   tip, ear tip — anything).
4. For each of the 7 direction slots: pose the limb in that direction
   (move whichever bones you'd like to record), then click the
   matching `Capture <direction>` button. The addon records the bone
   deltas relative to rest along with the end bone's resulting
   position.
5. At any frame, click **Save Pose** to record where the end bone is
   right now (in world space).
6. Scrub to a different frame, move the rig however you like (object
   transform, root pose-bone, both), then click **Restore Pose**. The
   addon solves a constrained least-squares blend over your captures,
   refines for slerp nonlinearity, and keyframes the result. The end
   bone lands at — or, if your captures don't span the target,
   clamps to the closest reachable point of — the saved world target.

## Presets

`View3D → Sidebar → Dummy Kin → Presets` exposes JSON save/load. Useful
for reusing setups across rigs that share bone names.

## License

GPL-2.0-or-later. See `LICENSE`.

## Credits

Concept and design by Jordan. Implementation written collaboratively
with Claude (Anthropic).
