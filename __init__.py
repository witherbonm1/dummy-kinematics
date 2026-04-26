# SPDX-License-Identifier: GPL-2.0-or-later
#
# Dummy Kinematics
# Concept and design by Jordan.
# Implementation written collaboratively with Claude (Anthropic).
#
# Pose-blending pseudo-IK: capture reference poses for each direction
# (Forward / Back / Left / Right / Up / Down / Retract), then blend them
# to approximate any target end-effector position. Not real IK — uses
# user-recorded poses as basis vectors, so output always looks natural.

"""
Dummy Kinematics — Blender 5.0+ addon
======================================

Pose-blending pseudo-IK system for fast, natural-feeling animation.

It's NOT real IK. It does not solve joint angles from an end-effector
position. Instead, the user records 7 hand-posed reference poses per
limb (Forward, Back, Left, Right, Up, Down, Retract), and the addon
*blends* those recordings to approximate any target position. Because
every input was authored by the animator, the output always looks
natural — knees bend the right way, hips rotate plausibly, no IK
flipping or pole-target headaches.

This pairs well with the Bone Lock addon: same Save/Restore concept,
but Restore intelligently blends recorded poses to land the end bone
near a target position, instead of forcing the exact saved transform.

Workflow
--------
1. **Setup a limb**
   - Pick an end bone (e.g. "RightFoot.L").
   - Name the limb ("Right Leg" / "Tail" / "Left Ear" — anything).
   - For each direction (Forward, Back, Left, Right, Up, Down, Retract):
     * Manually pose the limb that way using normal Blender pose tools.
     * Click "Capture <Direction>" — the addon records bone deltas from
       rest pose for ONLY the bones you actually touched.

2. **Use it**
   - Select the limb's end bone (or pick the limb in the panel).
   - Click "Save Pose" at frame 0 — records the end bone's current
     offset from the armature root.
   - Scrub to frame 40, move the armature however you want.
   - Click "Restore Pose" — the addon computes where the end bone needs
     to be (in armature-local space, so the pose travels with the rig),
     blends the captured directions to get there, and writes keyframes
     ONLY on bones that were touched during the relevant captures.

Out-of-reach behavior
---------------------
If the target is farther than the most-extended capture, the limb is
clamped to its maximum reach in that direction. It won't stretch
unnaturally — the capture set defines the limb's range.

What "shared direction names" means
------------------------------------
Every limb has the same 7 direction slots, but each limb's recordings
are completely independent. "Right Leg → Forward" and "Left Leg →
Forward" are different captures, as they should be.
"""

bl_info = {
    "name": "Dummy Kinematics",
    "author": "Jordan",
    "version": (1, 7, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Dummy Kin (Pose Mode)",
    "description": "Pose-blending pseudo-IK using user-recorded reference poses.",
    "category": "Animation",
}

import bpy
import json
import os
import numpy as np
from mathutils import Vector, Quaternion, Matrix
from bpy.types import Operator, Panel, PropertyGroup, UIList
from bpy.props import (
    StringProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
    IntProperty,
    FloatVectorProperty,
)


# ===========================================================================
# Constants
# ===========================================================================

CAPTURE_MODES = [
    ("SELECTION_ONLY", "Selection Only",
     "Record exactly the selected pose bones, even if they haven't moved from rest. "
     "Most predictable — you control exactly what's captured"),
    ("SELECTION_AND_MOVED", "Selection + Moved",
     "Record selected pose bones that ALSO differ from rest pose. "
     "Filters out accidentally-selected unmoved bones"),
    ("AUTO_DETECT", "Auto-Detect",
     "Record any bone that differs from rest pose, regardless of selection. "
     "Original behavior — convenient but can pick up unintended bones"),
]

# Preset filename suffix (.json files are auto-recognized as DK presets)
PRESET_FILE_EXT = ".json"
PRESET_FORMAT_VERSION = 1

DIRECTIONS = [
    ("FORWARD", "Forward", "+Y direction in armature-local space"),
    ("BACK",    "Back",    "-Y direction"),
    ("LEFT",    "Left",    "-X direction"),
    ("RIGHT",   "Right",   "+X direction"),
    ("UP",      "Up",      "+Z direction"),
    ("DOWN",    "Down",    "-Z direction"),
    ("RETRACT", "Retract", "Limb pulled in toward the body"),
]

# Threshold below which a captured rotation is considered "untouched" and
# excluded from the recording (so we don't keyframe bones the user didn't pose).
ROT_EPSILON = 1e-4
LOC_EPSILON = 1e-5
SCALE_EPSILON = 1e-4


# ===========================================================================
# Property data
# ===========================================================================

class DK_BoneDelta(PropertyGroup):
    """Per-bone delta from rest pose, captured as part of a reference pose.
    Only stored for bones the user actually touched."""
    bone_name: StringProperty()
    location: FloatVectorProperty(size=3)
    # Rotation stored as quaternion always (we convert from whatever the bone
    # uses) so the blend math is consistent.
    rotation: FloatVectorProperty(size=4, default=(1.0, 0.0, 0.0, 0.0))
    scale: FloatVectorProperty(size=3, default=(1.0, 1.0, 1.0))
    # Original rotation_mode of the pose bone, so we can write back correctly
    rotation_mode_at_capture: StringProperty(default="QUATERNION")


class DK_DirectionPose(PropertyGroup):
    """One captured reference pose for a limb in a specific direction."""
    direction: EnumProperty(items=DIRECTIONS, default="FORWARD")
    captured: BoolProperty(default=False)
    # End-bone offset from the armature root when this pose was captured.
    # Used as the basis vector for blending: "Forward" means whatever direction
    # the end bone went when the user captured Forward.
    end_offset: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0))
    bone_deltas: CollectionProperty(type=DK_BoneDelta)


def _on_end_bone_changed(self, context):
    """Auto-refresh rest_position whenever end_bone changes via any path
    (typing, prop_search dropdown, eyedropper, preset load). Without this,
    picking a bone via the search field leaves rest_position_valid=False
    and the blend math silently falls back to a wrong origin."""
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE" and obj.data == self.id_data:
        _refresh_rest_position(obj, self)


class DK_Limb(PropertyGroup):
    """A user-defined limb: end bone + 7 direction captures."""
    name: StringProperty(default="New Limb")
    end_bone: StringProperty(default="", update=_on_end_bone_changed)
    poses: CollectionProperty(type=DK_DirectionPose)

    # "Saved pose" target — armature-local end bone position recorded by
    # Save Pose, used by Restore Pose to drive the blend.
    saved_target: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0))
    saved_valid: BoolProperty(default=False)

    # Rest position of the end bone in armature-local space. Used as the
    # origin for direction calculations — without this, the blend math
    # treats armature-origin as "neutral" which is wildly wrong for any
    # bone that doesn't sit at the armature origin (e.g. a foot ~1m below).
    # Auto-populated when the user adds a limb or sets the end bone.
    rest_position: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0))
    rest_position_valid: BoolProperty(default=False)

    # Lock prevents accidental re-capture of references after the user is
    # happy with their setup. Capture buttons grey out when this is True.
    locked: BoolProperty(
        name="Lock Captures",
        default=False,
        description=("When locked, the direction capture buttons are disabled "
                     "to prevent accidentally overwriting your reference poses"),
    )

    def get_pose(self, direction: str) -> DK_DirectionPose | None:
        for p in self.poses:
            if p.direction == direction:
                return p
        return None

    def ensure_pose_slots(self):
        """Make sure all 7 direction slots exist (idempotent)."""
        existing = {p.direction for p in self.poses}
        for d_id, _, _ in DIRECTIONS:
            if d_id not in existing:
                p = self.poses.add()
                p.direction = d_id


class DK_ArmatureProps(PropertyGroup):
    """Per-armature data — stored on the armature data block."""
    limbs: CollectionProperty(type=DK_Limb)
    active_limb_index: IntProperty(default=0)

    # Capture behavior — toggleable so different workflows can be supported
    # without forcing one to win.
    capture_mode: EnumProperty(
        name="Capture Mode",
        items=CAPTURE_MODES,
        default="SELECTION_ONLY",
        description="What gets recorded when you click a direction capture button",
    )

    # UI fold-state
    show_settings: BoolProperty(
        name="Show Settings",
        default=False,
        description="Expand the settings sub-panel",
    )
    show_presets: BoolProperty(
        name="Show Presets",
        default=False,
        description="Expand the preset save/load sub-panel",
    )


# ===========================================================================
# Math helpers
# ===========================================================================

def _rest_pose_local(pb: bpy.types.PoseBone):
    """Return (loc, quat, scale) representing 'no offset from rest pose'."""
    return Vector((0, 0, 0)), Quaternion((1, 0, 0, 0)), Vector((1, 1, 1))


def _bone_current_delta(pb: bpy.types.PoseBone):
    """Get current pose-channel offset from rest as (loc, quat, scale)."""
    loc = pb.location.copy()
    if pb.rotation_mode == "QUATERNION":
        quat = pb.rotation_quaternion.copy()
    elif pb.rotation_mode == "AXIS_ANGLE":
        aa = pb.rotation_axis_angle
        # aa is (angle, x, y, z)
        quat = Quaternion(Vector((aa[1], aa[2], aa[3])), aa[0])
    else:  # Euler XYZ etc.
        quat = pb.rotation_euler.to_quaternion()
    sca = pb.scale.copy()
    return loc, quat, sca


def _apply_delta(pb: bpy.types.PoseBone, loc: Vector, quat: Quaternion, sca: Vector):
    """Write (loc, quat, scale) back into the pose bone in its native rotation
    mode."""
    pb.location = loc
    if pb.rotation_mode == "QUATERNION":
        pb.rotation_quaternion = quat
    elif pb.rotation_mode == "AXIS_ANGLE":
        axis = quat.axis
        angle = quat.angle
        pb.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    else:
        pb.rotation_euler = quat.to_euler(pb.rotation_mode)
    pb.scale = sca


def _delta_is_significant(loc: Vector, quat: Quaternion, sca: Vector) -> bool:
    """True if this delta represents a real user pose change, not just rest."""
    if loc.length > LOC_EPSILON:
        return True
    # Quaternion identity is (1,0,0,0). Compare via angle to identity.
    # quat.angle is the rotation angle — works regardless of sign convention.
    if abs(quat.angle) > ROT_EPSILON:
        return True
    if (abs(sca.x - 1.0) > SCALE_EPSILON or
        abs(sca.y - 1.0) > SCALE_EPSILON or
        abs(sca.z - 1.0) > SCALE_EPSILON):
        return True
    return False


def _end_bone_armature_local_pos(arm_obj, pb: bpy.types.PoseBone) -> Vector:
    """Position of the bone's head in armature-local space (i.e., independent
    of where the armature object is in the world). Using head rather than
    tail/origin matches what the user visually treats as 'where the foot is'."""
    return pb.matrix.to_translation()


def _bone_rest_local_pos(arm_obj, bone_name: str) -> Vector | None:
    """Where the bone's head sits in armature-local space when the rig is at
    rest (no pose deformation). Used as the origin for direction blending,
    so 'forward' means '+Y from rest position' rather than '+Y from origin'."""
    bone = arm_obj.data.bones.get(bone_name)
    if bone is None:
        return None
    # bone.matrix_local is the bone's rest matrix in armature-local space.
    return bone.matrix_local.to_translation()


def _refresh_rest_position(arm_obj, limb: DK_Limb):
    """Update the limb's stored rest_position from the current end_bone."""
    if not limb.end_bone:
        limb.rest_position_valid = False
        return
    rp = _bone_rest_local_pos(arm_obj, limb.end_bone)
    if rp is None:
        limb.rest_position_valid = False
        return
    limb.rest_position = rp
    limb.rest_position_valid = True


def _quat_slerp_weighted(quats_and_weights):
    """Iteratively slerp a list of (quaternion, weight) pairs into one
    blended quaternion. Weights need not sum to 1 — they're normalized.

    Two design choices that prevent "long-way-around" blends:
      1. Items are processed in DESCENDING weight order, so the heaviest
         quaternion anchors the hemisphere choice. Otherwise a small-
         weight outlier processed first can park the accumulator on the
         wrong side of the hypersphere, and every subsequent rotation
         lerps the long way to align with that bad reference.
      2. All quaternions are pre-aligned (sign-flipped if needed) to the
         anchor BEFORE slerping, not lazily against the moving
         accumulator. This keeps later iterations from drifting hemisphere
         as the accumulator rotates."""
    items = [(q, w) for q, w in quats_and_weights if w > 1e-9]
    if not items:
        return Quaternion((1, 0, 0, 0))
    if len(items) == 1:
        return items[0][0].copy()

    total_w = sum(w for _, w in items)
    if total_w < 1e-9:
        return Quaternion((1, 0, 0, 0))

    # Heaviest first — its hemisphere becomes the anchor.
    items.sort(key=lambda iw: -iw[1])
    anchor = items[0][0].copy()
    aligned = []
    for q, w in items:
        if anchor.dot(q) < 0:
            aligned.append((Quaternion((-q.w, -q.x, -q.y, -q.z)), w))
        else:
            aligned.append((q.copy(), w))

    q_acc = aligned[0][0]
    w_acc = aligned[0][1] / total_w
    for q, w in aligned[1:]:
        wn = w / total_w
        if w_acc + wn < 1e-9:
            continue
        t = wn / (w_acc + wn)
        q_acc = q_acc.slerp(q, t)
        w_acc += wn

    q_acc.normalize()
    return q_acc


# ===========================================================================
# Capture & restore core
# ===========================================================================

def _capture_pose_for_limb(arm_obj, limb: DK_Limb, direction: str,
                           capture_mode: str,
                           selected_bone_names: set[str] | None = None) -> int:
    """Snapshot bones' current delta from rest into the named direction slot.

    Which bones get recorded depends on capture_mode:
      - SELECTION_ONLY: every selected bone, even if at rest
      - SELECTION_AND_MOVED: selected bones that also differ from rest
      - AUTO_DETECT: any bone that differs from rest, regardless of selection

    Returns the number of bones recorded."""
    pose = limb.get_pose(direction)
    if pose is None:
        return 0
    pose.bone_deltas.clear()

    end_pb = arm_obj.pose.bones.get(limb.end_bone)
    if end_pb is None:
        return 0
    pose.end_offset = _end_bone_armature_local_pos(arm_obj, end_pb)

    if selected_bone_names is None:
        selected_bone_names = set()

    count = 0
    for pb in arm_obj.pose.bones:
        loc, quat, sca = _bone_current_delta(pb)
        is_selected = pb.name in selected_bone_names
        is_moved = _delta_is_significant(loc, quat, sca)

        # Decide whether to record this bone based on capture mode
        if capture_mode == "SELECTION_ONLY":
            if not is_selected:
                continue
        elif capture_mode == "SELECTION_AND_MOVED":
            if not (is_selected and is_moved):
                continue
        elif capture_mode == "AUTO_DETECT":
            if not is_moved:
                continue
        else:
            # Unknown mode — default to safest behavior (selection + moved)
            if not (is_selected and is_moved):
                continue

        d = pose.bone_deltas.add()
        d.bone_name = pb.name
        d.location = loc
        d.rotation = (quat.w, quat.x, quat.y, quat.z)
        d.scale = sca
        d.rotation_mode_at_capture = pb.rotation_mode
        count += 1

    pose.captured = True
    return count


def _evaluate_blended_end_position(arm_obj, limb: DK_Limb,
                                   weights: dict) -> Vector | None:
    """Predict where the end bone's HEAD will land in armature-local space
    if `weights` were applied to the limb. Walks the chain analytically,
    composing blended pose channels at each bone — does NOT touch the rig
    or trigger depsgraph eval, so it's safe to call repeatedly inside an
    operator without fighting keyframes.

    This is the nonlinear ground-truth that the LS solver's linear
    cap_off-based prediction is approximating. Used by Restore's
    refinement loop to correct for slerp-arc-vs-line error."""
    end_pb = arm_obj.pose.bones.get(limb.end_bone)
    if end_pb is None:
        return None

    captured_bones = set()
    for pose in limb.poses:
        if pose.captured:
            for d in pose.bone_deltas:
                captured_bones.add(d.bone_name)

    chain = [end_pb]
    cur = end_pb.parent
    while cur is not None and cur.name in captured_bones:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    anchor = cur

    if anchor is None:
        m = Matrix.Identity(4)
        prev_rest = Matrix.Identity(4)
    else:
        m = anchor.matrix.copy()
        prev_rest = anchor.bone.matrix_local

    rest_weight = weights.get("_REST", 0.0)

    for pb in chain:
        # Blend pose channels for this bone (same logic as _apply_blended_pose,
        # but pure — doesn't write to the bone).
        loc_acc = Vector((0, 0, 0))
        sca_acc = Vector((0, 0, 0))
        quats = []
        total_w = 0.0
        for d_id, w in weights.items():
            if d_id == "_REST" or w <= 1e-9:
                continue
            pose = limb.get_pose(d_id)
            if pose is None or not pose.captured:
                continue
            delta = None
            for d in pose.bone_deltas:
                if d.bone_name == pb.name:
                    delta = d
                    break
            if delta is not None:
                loc = Vector(delta.location)
                q = Quaternion(delta.rotation)
                sca = Vector(delta.scale)
            else:
                loc = Vector((0, 0, 0))
                q = Quaternion((1, 0, 0, 0))
                sca = Vector((1, 1, 1))
            loc_acc += loc * w
            sca_acc += sca * w
            quats.append((q, w))
            total_w += w

        if rest_weight > 1e-9:
            sca_acc += Vector((1, 1, 1)) * rest_weight
            quats.append((Quaternion((1, 0, 0, 0)), rest_weight))
            total_w += rest_weight

        if total_w < 1e-9:
            loc_b = Vector((0, 0, 0))
            quat_b = Quaternion((1, 0, 0, 0))
            sca_b = Vector((1, 1, 1))
        else:
            loc_b = loc_acc / total_w
            sca_b = sca_acc / total_w
            quat_b = _quat_slerp_weighted(quats)

        # Compose this bone's pose-basis matrix (loc * rot * scale, in
        # bone-local space, Blender convention).
        loc_m = Matrix.Translation(loc_b)
        rot_m = quat_b.to_matrix().to_4x4()
        sca_m = Matrix.Identity(4)
        sca_m[0][0] = sca_b.x
        sca_m[1][1] = sca_b.y
        sca_m[2][2] = sca_b.z
        basis = loc_m @ rot_m @ sca_m

        bone_rest = pb.bone.matrix_local
        offset = prev_rest.inverted() @ bone_rest
        m = m @ offset @ basis
        prev_rest = bone_rest

    return m.to_translation()


def _compute_chain_rest_pos(arm_obj, limb: DK_Limb) -> Vector | None:
    """Compute where the end bone's HEAD sits in armature-local space if the
    limb's captured bones were at rest, but every ancestor above the limb is
    in its current posed state.

    This is the displacement origin for the blend. If a root/hip pose-bone
    moves the rig, this picks up that movement; if only the limb itself is
    posed (stale keyframes from a previous bad Restore), this ignores it.

    Strategy: find the first ancestor that isn't part of the limb's captures
    (the 'anchor'). Take its current pose-matrix (which reflects every
    ancestor above it). Multiply down through rest-matrix offsets to the
    end bone. No depsgraph eval needed — purely from data, so existing
    keyframes on limb bones can't poison the answer."""
    end_pb = arm_obj.pose.bones.get(limb.end_bone)
    if end_pb is None:
        return None

    # Set of bones the limb's captures will overwrite — these are the bones
    # whose CURRENT pose state we want to ignore when computing the origin.
    limb_bones = set()
    for pose in limb.poses:
        if not pose.captured:
            continue
        for d in pose.bone_deltas:
            limb_bones.add(d.bone_name)

    # Walk up the parent chain, collecting captured ancestors of end_pb until
    # we hit a non-captured bone (the anchor) or run out of parents.
    chain_to_anchor = [end_pb]
    cur = end_pb.parent
    while cur is not None and cur.name in limb_bones:
        chain_to_anchor.append(cur)
        cur = cur.parent
    chain_to_anchor.reverse()
    # chain_to_anchor[-1] is end_pb. chain_to_anchor[0] is the topmost
    # captured bone in the chain (often the same as end_pb if only foot
    # was captured).

    topmost = chain_to_anchor[0]
    anchor = topmost.parent  # may be None if topmost is root

    if anchor is None:
        # No anchor — chain reaches the armature root. The stable frame is
        # armature space, so build from the topmost bone's REST matrix.
        m = topmost.bone.matrix_local.copy()
        prev_rest = m
        for i in range(1, len(chain_to_anchor)):
            child_rest = chain_to_anchor[i].bone.matrix_local
            m = m @ (prev_rest.inverted() @ child_rest)
            prev_rest = child_rest
        return m.to_translation()

    # Start at anchor's CURRENT pose matrix (already includes everything
    # above the anchor, including any root/hip translation the user posed).
    m = anchor.matrix.copy()
    prev_rest = anchor.bone.matrix_local
    for pb in chain_to_anchor:
        child_rest = pb.bone.matrix_local
        m = m @ (prev_rest.inverted() @ child_rest)
        prev_rest = child_rest
    return m.to_translation()


def _solve_box_constrained_ls(A: np.ndarray, d: np.ndarray,
                              max_iter: int = 80, tol: float = 1e-7) -> np.ndarray:
    """Solve  min ||A w - d||^2   subject to  0 <= w_i <= 1.

    A is 3xN (basis vectors as columns), d is 3-vector. Returns N weights.

    Method: projected gradient descent with step = 1 / spectral_norm(A^T A).
    Reliable for small N (we have at most 6) and converges in well under
    100 iterations even on near-singular bases. No scipy dependency."""
    n = A.shape[1]
    if n == 0:
        return np.zeros(0)

    AtA = A.T @ A
    Atd = A.T @ d

    # Step size: inverse Lipschitz constant of the gradient
    eigmax = float(np.linalg.eigvalsh(AtA).max()) if n > 0 else 0.0
    if eigmax < 1e-12:
        return np.zeros(n)
    lr = 1.0 / eigmax

    w = np.zeros(n)
    for _ in range(max_iter):
        grad = AtA @ w - Atd
        w_new = np.clip(w - lr * grad, 0.0, 1.0)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w


def _compute_blend_weights(limb: DK_Limb, target_local: Vector,
                           current_chain_rest: Vector | None = None) -> dict[str, float]:
    """Given a target end-bone position in armature-local space, compute how
    much of each captured direction's pose to mix in.

    Strategy (v3 — least-squares solve over capture basis):
    - Each captured direction's recorded chain-delta vector becomes a basis
      vector. We do NOT assume the captures align with armature axes —
      they're whatever poses the user authored.
    - We solve for a weight per direction in [0, 1] that minimizes the
      distance between (sum of weighted capture-deltas) and the actual
      displacement we need. Box constraints prevent extrapolation past
      the captured extents.
    - The leftover weight (1 - sum_of_directional, clamped >= 0) blends
      toward rest pose via the special "_REST" key.
    - Retract is excluded from the basis (it has no associated direction);
      the addon could expose it as a manual blend later if needed.

    This is the closest thing DK does to actual IK: it asks "what blend of
    my captured poses lands the foot closest to the target?" rather than
    relying on fixed axes that won't match every rig orientation."""
    if limb.rest_position_valid:
        rest_origin = Vector(limb.rest_position)
    else:
        retract = limb.get_pose("RETRACT")
        if retract is not None and retract.captured:
            rest_origin = Vector(retract.end_offset)
        else:
            rest_origin = Vector((0, 0, 0))

    displacement_origin = (Vector(current_chain_rest)
                           if current_chain_rest is not None
                           else rest_origin)

    displacement = Vector(target_local) - displacement_origin

    weights: dict[str, float] = {d_id: 0.0 for d_id, _, _ in DIRECTIONS}

    # Build basis matrix A (3 x N) from captured directions, in the order
    # they appear in DIRECTIONS so we can map columns back to direction ids.
    # Retract is included now — under the LS solver, every capture is just
    # a 3D direction in arm-local space, no axis assumption needed. If the
    # Retract pose (e.g. knee-fully-bent) helps reach an intermediate target
    # better than the cardinal captures, the solver will weight it in.
    basis_dirs: list[str] = []
    basis_vecs: list[list[float]] = []
    for d_id, _, _ in DIRECTIONS:
        pose = limb.get_pose(d_id)
        if pose is None or not pose.captured:
            continue
        cap_off = Vector(pose.end_offset) - rest_origin
        # Skip degenerate captures (no measurable end-bone movement)
        if cap_off.length < 1e-6:
            continue
        basis_dirs.append(d_id)
        basis_vecs.append([cap_off.x, cap_off.y, cap_off.z])

    if not basis_dirs:
        # No usable captures — everything to rest.
        weights["_REST"] = 1.0
        return weights

    A = np.asarray(basis_vecs, dtype=float).T  # 3 x N
    d = np.asarray([displacement.x, displacement.y, displacement.z], dtype=float)

    w = _solve_box_constrained_ls(A, d)

    total = 0.0
    for i, d_id in enumerate(basis_dirs):
        wi = float(max(0.0, min(1.0, w[i])))
        weights[d_id] = wi
        total += wi

    weights["_REST"] = max(0.0, 1.0 - total)
    return weights


def _apply_blended_pose(arm_obj, limb: DK_Limb, weights: dict[str, float],
                       insert_keyframes: bool, frame: int):
    """Blend all captured poses by their weights and write the result onto
    the armature. Only bones that appear in at least one captured pose with
    nonzero weight get touched."""
    # Collect every bone that appears in any contributing captured pose.
    # _REST is a virtual pose (identity deltas) — it contributes weight to
    # the blend but doesn't define which bones are touched.
    rest_weight = weights.get("_REST", 0.0)
    contributing_poses = []
    bone_names_set = set()
    for d_id, w in weights.items():
        if d_id == "_REST" or w <= 1e-9:
            continue
        pose = limb.get_pose(d_id)
        if pose is None or not pose.captured:
            continue
        contributing_poses.append((pose, w))
        for d in pose.bone_deltas:
            bone_names_set.add(d.bone_name)

    # If only rest_weight is nonzero (target == rest), no bones are touched.
    # That's the correct no-op: target at rest = no movement.
    if not contributing_poses:
        return 0

    # For each bone, gather its recorded delta from each contributing pose
    # (or rest if absent in that pose), then blend.
    touched = 0
    for bname in bone_names_set:
        pb = arm_obj.pose.bones.get(bname)
        if pb is None:
            continue

        loc_acc = Vector((0, 0, 0))
        sca_acc = Vector((0, 0, 0))
        quats_for_slerp = []
        total_w = 0.0

        for pose, w in contributing_poses:
            # Find this bone's delta in this pose; if not recorded → rest.
            delta = None
            for d in pose.bone_deltas:
                if d.bone_name == bname:
                    delta = d
                    break

            if delta is not None:
                loc = Vector(delta.location)
                q = Quaternion(delta.rotation)  # (w, x, y, z)
                sca = Vector(delta.scale)
            else:
                loc = Vector((0, 0, 0))
                q = Quaternion((1, 0, 0, 0))
                sca = Vector((1, 1, 1))

            loc_acc += loc * w
            sca_acc += sca * w
            quats_for_slerp.append((q, w))
            total_w += w

        # Blend toward rest for the leftover weight. Identity quat, zero
        # location, unit scale — exactly the bone's rest pose. Without this
        # the blend re-normalizes to total_directional and a half-Forward
        # target would produce full-Forward.
        if rest_weight > 1e-9:
            sca_acc += Vector((1, 1, 1)) * rest_weight
            quats_for_slerp.append((Quaternion((1, 0, 0, 0)), rest_weight))
            total_w += rest_weight

        if total_w < 1e-9:
            continue

        loc_blended = loc_acc / total_w
        sca_blended = sca_acc / total_w
        quat_blended = _quat_slerp_weighted(quats_for_slerp)

        _apply_delta(pb, loc_blended, quat_blended, sca_blended)

        if insert_keyframes:
            try:
                pb.keyframe_insert("location", frame=frame)
                if pb.rotation_mode == "QUATERNION":
                    pb.keyframe_insert("rotation_quaternion", frame=frame)
                elif pb.rotation_mode == "AXIS_ANGLE":
                    pb.keyframe_insert("rotation_axis_angle", frame=frame)
                else:
                    pb.keyframe_insert("rotation_euler", frame=frame)
                pb.keyframe_insert("scale", frame=frame)
            except RuntimeError:
                # keyframe_insert can fail if action is locked; skip silently
                pass

        touched += 1

    return touched


# ===========================================================================
# Operators — Limb management
# ===========================================================================

class DK_OT_add_limb(Operator):
    bl_idname = "dk.add_limb"
    bl_label = "Add Limb"
    bl_description = "Create a new limb entry on the active armature"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def execute(self, context):
        arm_data = context.active_object.data
        dk = arm_data.dummy_kin
        limb = dk.limbs.add()

        # Display name is always "Limb N" — separate from the bone target,
        # so the user understands these are two different things.
        # Find next available number (handles the case where some limbs
        # were removed and we don't want collisions).
        existing_names = {lb.name for lb in dk.limbs if lb is not limb}
        n = len(dk.limbs)
        candidate = f"Limb {n}"
        while candidate in existing_names:
            n += 1
            candidate = f"Limb {n}"
        limb.name = candidate

        # If a pose bone is currently active, use it as the end-bone target
        # only — NOT as the display name.
        apb = context.active_pose_bone
        if apb is not None:
            limb.end_bone = apb.name
            _refresh_rest_position(context.active_object, limb)

        limb.ensure_pose_slots()
        dk.active_limb_index = len(dk.limbs) - 1
        return {"FINISHED"}


class DK_OT_remove_limb(Operator):
    bl_idname = "dk.remove_limb"
    bl_label = "Remove Limb"
    bl_description = "Delete the active limb and all its captured poses"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        dk = context.active_object.data.dummy_kin
        dk.limbs.remove(dk.active_limb_index)
        dk.active_limb_index = max(0, dk.active_limb_index - 1)
        return {"FINISHED"}


class DK_OT_set_end_bone_from_active(Operator):
    bl_idname = "dk.set_end_bone_from_active"
    bl_label = "Use Active Bone"
    bl_description = "Set this limb's end bone to the currently selected pose bone"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        if context.active_pose_bone is None:
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        dk = context.active_object.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        limb.end_bone = context.active_pose_bone.name
        _refresh_rest_position(context.active_object, limb)
        return {"FINISHED"}


class DK_OT_refresh_rest(Operator):
    bl_idname = "dk.refresh_rest"
    bl_label = "Refresh Rest Position"
    bl_description = ("Recompute the limb's rest-pose origin from the current "
                      "end bone. Run this after changing the end bone or after "
                      "edit-bone changes")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        if not (0 <= dk.active_limb_index < len(dk.limbs)):
            return False
        return bool(dk.limbs[dk.active_limb_index].end_bone)

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        _refresh_rest_position(arm_obj, limb)
        if limb.rest_position_valid:
            self.report({"INFO"},
                        f"Rest position: ({limb.rest_position[0]:.3f}, "
                        f"{limb.rest_position[1]:.3f}, "
                        f"{limb.rest_position[2]:.3f})")
        else:
            self.report({"WARNING"}, "Couldn't compute rest position — "
                                     "is the end bone valid?")
        return {"FINISHED"}


class DK_OT_toggle_lock(Operator):
    bl_idname = "dk.toggle_lock"
    bl_label = "Toggle Lock"
    bl_description = "Lock or unlock this limb's captures to prevent accidental changes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        dk = context.active_object.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        limb.locked = not limb.locked
        self.report({"INFO"},
                    f"'{limb.name}' {'locked' if limb.locked else 'unlocked'}")
        return {"FINISHED"}


# ===========================================================================
# Operators — Capture / Clear directions
# ===========================================================================

class DK_OT_capture_direction(Operator):
    bl_idname = "dk.capture_direction"
    bl_label = "Capture Direction"
    bl_description = ("Record current pose into this limb's chosen direction "
                      "slot. Which bones get recorded depends on Capture Mode "
                      "in Settings")
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=DIRECTIONS, default="FORWARD")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        if context.mode != "POSE":
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        if not limb.end_bone or limb.end_bone not in arm_obj.pose.bones:
            self.report({"ERROR"}, "Limb has no valid end bone — set one first")
            return {"CANCELLED"}
        if limb.locked:
            self.report({"ERROR"},
                        "This limb is locked. Click Unlock in the panel to "
                        "modify captures")
            return {"CANCELLED"}
        limb.ensure_pose_slots()

        # Make sure rest position is current — captures need it as the
        # origin point for blending. Cheap to refresh every capture.
        _refresh_rest_position(arm_obj, limb)

        selected = {pb.name for pb in (context.selected_pose_bones or [])}
        n = _capture_pose_for_limb(arm_obj, limb, self.direction,
                                   capture_mode=dk.capture_mode,
                                   selected_bone_names=selected)
        label = next((lbl for d, lbl, _ in DIRECTIONS if d == self.direction),
                     self.direction)
        mode_label = next((lbl for k, lbl, _ in CAPTURE_MODES
                           if k == dk.capture_mode), dk.capture_mode)
        if n == 0:
            # Custom guidance per mode — different reasons for empty captures
            if dk.capture_mode == "SELECTION_ONLY":
                hint = "select the bones you want to record first"
            elif dk.capture_mode == "SELECTION_AND_MOVED":
                hint = "select AND pose the bones you want to record"
            else:
                hint = "pose some bones away from rest first"
            self.report({"WARNING"},
                        f"{label}: nothing recorded ({mode_label}) — {hint}")
        else:
            self.report({"INFO"},
                        f"{label}: captured {n} bone(s) for '{limb.name}' "
                        f"({mode_label})")
        return {"FINISHED"}


class DK_OT_clear_direction(Operator):
    bl_idname = "dk.clear_direction"
    bl_label = "Clear Direction"
    bl_description = "Discard this direction's captured pose"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=DIRECTIONS, default="FORWARD")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        dk = context.active_object.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        if limb.locked:
            self.report({"ERROR"},
                        "This limb is locked. Click Unlock in the panel to "
                        "modify captures")
            return {"CANCELLED"}
        pose = limb.get_pose(self.direction)
        if pose is not None:
            pose.bone_deltas.clear()
            pose.captured = False
            pose.end_offset = (0.0, 0.0, 0.0)
        return {"FINISHED"}


# ===========================================================================
# Operators — Save / Restore (the actual workflow)
# ===========================================================================

class DK_OT_save_pose(Operator):
    bl_idname = "dk.save_pose"
    bl_label = "Save Pose"
    bl_description = ("Record the active limb's end-bone position (in "
                      "armature-local space) so Restore Pose can blend "
                      "captures to reach it later")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        if not (0 <= dk.active_limb_index < len(dk.limbs)):
            return False
        limb = dk.limbs[dk.active_limb_index]
        return bool(limb.end_bone) and limb.end_bone in obj.pose.bones

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        end_pb = arm_obj.pose.bones[limb.end_bone]
        _refresh_rest_position(arm_obj, limb)
        # Save in WORLD space so the target stays planted when the armature
        # moves. Restore converts back to armature-local for the blend math.
        local = _end_bone_armature_local_pos(arm_obj, end_pb)
        limb.saved_target = arm_obj.matrix_world @ local
        limb.saved_valid = True
        self.report({"INFO"},
                    f"Saved target for '{limb.name}' at "
                    f"{tuple(round(x, 3) for x in limb.saved_target)}")
        return {"FINISHED"}


class DK_OT_restore_pose(Operator):
    bl_idname = "dk.restore_pose"
    bl_label = "Restore Pose"
    bl_description = ("Blend captured directions to move the limb's end bone "
                      "toward the saved target. Optionally keyframes the "
                      "result on the current frame")
    bl_options = {"REGISTER", "UNDO"}

    insert_keyframes: BoolProperty(
        name="Insert Keyframes",
        default=True,
        description="Keyframe the blended result on the current frame",
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        if not (0 <= dk.active_limb_index < len(dk.limbs)):
            return False
        limb = dk.limbs[dk.active_limb_index]
        if not limb.saved_valid:
            return False
        # Need at least one captured pose to blend
        return any(p.captured for p in limb.poses)

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]

        _refresh_rest_position(arm_obj, limb)

        # saved_target is in world space; convert to armature-local using the
        # armature's current world matrix.
        target_world = Vector(limb.saved_target)
        target = arm_obj.matrix_world.inverted() @ target_world

        # Compute the chain-driven origin analytically (no depsgraph eval,
        # so existing keyframes on limb bones can't corrupt the reading).
        end_pb = arm_obj.pose.bones.get(limb.end_bone)
        if end_pb is None:
            self.report({"ERROR"}, "End bone missing on this armature")
            return {"CANCELLED"}
        chain_rest = _compute_chain_rest_pos(arm_obj, limb)
        if chain_rest is None:
            chain_rest = Vector(limb.rest_position)

        weights = _compute_blend_weights(limb, target,
                                         current_chain_rest=chain_rest)

        # --- Iterative refinement -----------------------------------------
        # The LS solver minimizes |sum(wᵢ·cap_offᵢ) - displacement| under
        # the LINEAR assumption that pose blending sums cap_offs. Actual
        # pose blending uses slerp, which traces an arc — so the foot lands
        # off the linear prediction. We compensate by analytically
        # evaluating the actual landing point, then nudging the target by
        # a damped fraction of the error and re-solving.
        # Damping (step < 1.0) avoids the overshoot/oscillation you get
        # with a full-step update when the nonlinearity is strong, which
        # happens for limbs with large rotation deltas between captures.
        # We track the best (lowest-error) solution seen so we never
        # return a solution worse than what we started with.
        refine_log = []
        target_eff = Vector(target)
        damping = 0.5

        # Evaluate the initial LS solution
        actual = _evaluate_blended_end_position(arm_obj, limb, weights)
        best_err_len = (Vector(target) - actual).length if actual is not None else float("inf")
        best_weights = {k: v for k, v in weights.items()}
        refine_log.append((0, best_err_len))

        max_refine = 8
        for refine_iter in range(1, max_refine + 1):
            if best_err_len < 1e-3:
                break
            err = Vector(target) - (actual if actual is not None else Vector(target))
            target_eff = target_eff + err * damping
            new_weights = _compute_blend_weights(
                limb, target_eff, current_chain_rest=chain_rest)
            new_actual = _evaluate_blended_end_position(arm_obj, limb, new_weights)
            if new_actual is None:
                break
            new_err_len = (Vector(target) - new_actual).length
            refine_log.append((refine_iter, new_err_len))
            if new_err_len < best_err_len:
                best_err_len = new_err_len
                best_weights = {k: v for k, v in new_weights.items()}
                weights = new_weights
                actual = new_actual
            else:
                # Step didn't help — shrink damping and try again from
                # current best, instead of accepting a worse solution.
                damping *= 0.5
                if damping < 1e-3:
                    break

        weights = best_weights
        # ------------------------------------------------------------------

        # Quick sanity: do we have any contributing weight at all?
        total = sum(weights.values())
        if total < 1e-6:
            self.report({"WARNING"},
                        "No captured directions match the target; nothing to blend. "
                        "Capture some directions first")
            return {"CANCELLED"}

        frame = context.scene.frame_current
        n = _apply_blended_pose(arm_obj, limb, weights,
                                insert_keyframes=self.insert_keyframes,
                                frame=frame)

        # Build a friendly mix summary
        mix_parts = [f"{lbl} {weights[d]*100:.0f}%"
                     for d, lbl, _ in DIRECTIONS
                     if weights.get(d, 0.0) > 0.01]
        rest_w = weights.get("_REST", 0.0)
        if rest_w > 0.01:
            mix_parts.append(f"Rest {rest_w*100:.0f}%")
        mix_str = ", ".join(mix_parts) if mix_parts else "no significant mix"
        kf_str = f" (keyframed @ frame {frame})" if self.insert_keyframes else ""
        self.report({"INFO"},
                    f"Restored '{limb.name}': {n} bones, mix = {mix_str}{kf_str}")
        return {"FINISHED"}


class DK_OT_clear_saved(Operator):
    bl_idname = "dk.clear_saved"
    bl_label = "Clear Saved Target"
    bl_description = "Discard the saved target position for this limb"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        return 0 <= dk.active_limb_index < len(dk.limbs)

    def execute(self, context):
        dk = context.active_object.data.dummy_kin
        limb = dk.limbs[dk.active_limb_index]
        limb.saved_valid = False
        limb.saved_target = (0.0, 0.0, 0.0)
        return {"FINISHED"}


# ===========================================================================
# Operators — Preset Save / Load (JSON)
# ===========================================================================

def _limb_to_dict(limb: DK_Limb) -> dict:
    """Serialize a limb (and all its captures) to a JSON-safe dict."""
    out = {
        "name": limb.name,
        "end_bone": limb.end_bone,
        "rest_position": list(limb.rest_position),
        "rest_position_valid": bool(limb.rest_position_valid),
        "locked": bool(limb.locked),
        "poses": [],
    }
    for pose in limb.poses:
        if not pose.captured:
            continue
        out["poses"].append({
            "direction": pose.direction,
            "end_offset": list(pose.end_offset),
            "bone_deltas": [
                {
                    "bone_name": d.bone_name,
                    "location": list(d.location),
                    "rotation": list(d.rotation),
                    "scale": list(d.scale),
                    "rotation_mode_at_capture": d.rotation_mode_at_capture,
                }
                for d in pose.bone_deltas
            ],
        })
    return out


def _dict_to_limb(data: dict, limb: DK_Limb, target_armature: bpy.types.Object,
                  match_stats: dict):
    """Populate a freshly-added limb from a serialized dict.

    match_stats is a dict that gets mutated with per-load reporting:
      - matched_bones: int
      - missing_bones: set[str]
      - matched_end_bone: bool
    """
    limb.name = data.get("name", "Loaded Limb")
    end_bone = data.get("end_bone", "")
    limb.end_bone = end_bone

    target_bone_names = {b.name for b in target_armature.data.bones}

    if end_bone and end_bone not in target_bone_names:
        match_stats["missing_bones"].add(end_bone)
        match_stats["matched_end_bone"] = False
    else:
        match_stats["matched_end_bone"] = True

    # Restore rest position if present in the file. If not (e.g. older preset),
    # try to compute it from the bone — only works if bone names match.
    rp = data.get("rest_position")
    rp_valid = data.get("rest_position_valid", False)
    if rp is not None and rp_valid:
        limb.rest_position = (float(rp[0]), float(rp[1]), float(rp[2]))
        limb.rest_position_valid = True
    else:
        _refresh_rest_position(target_armature, limb)

    limb.locked = bool(data.get("locked", False))

    limb.ensure_pose_slots()

    for pose_data in data.get("poses", []):
        direction = pose_data.get("direction")
        pose = limb.get_pose(direction)
        if pose is None:
            continue   # Unknown direction in the file — skip silently
        pose.captured = True
        eo = pose_data.get("end_offset", [0.0, 0.0, 0.0])
        pose.end_offset = (float(eo[0]), float(eo[1]), float(eo[2]))
        for d_data in pose_data.get("bone_deltas", []):
            bname = d_data.get("bone_name", "")
            if not bname:
                continue
            if bname not in target_bone_names:
                # Skip bones the target armature doesn't have, but tally for reporting
                match_stats["missing_bones"].add(bname)
                continue
            d = pose.bone_deltas.add()
            d.bone_name = bname
            loc = d_data.get("location", [0, 0, 0])
            rot = d_data.get("rotation", [1, 0, 0, 0])
            sca = d_data.get("scale", [1, 1, 1])
            d.location = (float(loc[0]), float(loc[1]), float(loc[2]))
            d.rotation = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
            d.scale = (float(sca[0]), float(sca[1]), float(sca[2]))
            d.rotation_mode_at_capture = d_data.get("rotation_mode_at_capture",
                                                    "QUATERNION")
            match_stats["matched_bones"] += 1


class DK_OT_save_preset(Operator):
    bl_idname = "dk.save_preset"
    bl_label = "Save DK Preset"
    bl_description = ("Save all limbs and captures from this armature to a "
                      "JSON file, so they can be loaded onto another armature "
                      "with matching bone names")
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = PRESET_FILE_EXT
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    save_all: BoolProperty(
        name="Save All Limbs",
        default=True,
        description="If false, only the active limb is saved",
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        dk = obj.data.dummy_kin
        return len(dk.limbs) > 0

    def invoke(self, context, event):
        # Suggest a filename if user hasn't picked one yet
        if not self.filepath:
            arm_name = context.active_object.data.name
            self.filepath = bpy.path.clean_name(arm_name) + "_DK" + PRESET_FILE_EXT
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin

        if self.save_all:
            limbs_to_save = list(dk.limbs)
        else:
            if not (0 <= dk.active_limb_index < len(dk.limbs)):
                self.report({"ERROR"}, "No active limb to save")
                return {"CANCELLED"}
            limbs_to_save = [dk.limbs[dk.active_limb_index]]

        payload = {
            "format": "DummyKinematics",
            "version": PRESET_FORMAT_VERSION,
            "source_armature": arm_obj.data.name,
            "limbs": [_limb_to_dict(lb) for lb in limbs_to_save],
        }

        # Make sure the path has the right extension
        path = self.filepath
        if not path.lower().endswith(PRESET_FILE_EXT):
            path += PRESET_FILE_EXT

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to write preset: {e}")
            return {"CANCELLED"}

        self.report({"INFO"},
                    f"Saved {len(limbs_to_save)} limb(s) to "
                    f"{os.path.basename(path)}")
        return {"FINISHED"}


class DK_OT_load_preset(Operator):
    bl_idname = "dk.load_preset"
    bl_label = "Load DK Preset"
    bl_description = ("Load limbs and captures from a JSON preset file onto "
                      "this armature. Bones missing on the target are skipped "
                      "and reported")
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filename_ext = PRESET_FILE_EXT
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    replace_existing: BoolProperty(
        name="Replace Existing Limbs",
        default=False,
        description=("If enabled, all current limbs on this armature are "
                     "deleted before loading. Otherwise loaded limbs are "
                     "appended"),
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to read preset: {e}")
            return {"CANCELLED"}
        except json.JSONDecodeError as e:
            self.report({"ERROR"}, f"Preset file is not valid JSON: {e}")
            return {"CANCELLED"}

        if payload.get("format") != "DummyKinematics":
            self.report({"ERROR"},
                        "File doesn't look like a Dummy Kinematics preset")
            return {"CANCELLED"}

        version = payload.get("version", 0)
        if version > PRESET_FORMAT_VERSION:
            self.report({"WARNING"},
                        f"Preset format version {version} is newer than this "
                        f"addon supports ({PRESET_FORMAT_VERSION}); some data "
                        f"may not load correctly")

        if self.replace_existing:
            dk.limbs.clear()

        match_stats = {
            "matched_bones": 0,
            "missing_bones": set(),
            "limbs_loaded": 0,
            "limbs_with_missing_end_bone": [],
        }

        for limb_data in payload.get("limbs", []):
            new_limb = dk.limbs.add()
            sub_stats = {
                "matched_bones": 0,
                "missing_bones": set(),
                "matched_end_bone": True,
            }
            _dict_to_limb(limb_data, new_limb, arm_obj, sub_stats)
            match_stats["matched_bones"] += sub_stats["matched_bones"]
            match_stats["missing_bones"].update(sub_stats["missing_bones"])
            if not sub_stats["matched_end_bone"]:
                match_stats["limbs_with_missing_end_bone"].append(new_limb.name)
            match_stats["limbs_loaded"] += 1

        if match_stats["limbs_loaded"] > 0:
            dk.active_limb_index = len(dk.limbs) - 1

        # Report results
        n_missing = len(match_stats["missing_bones"])
        n_matched = match_stats["matched_bones"]
        n_limbs = match_stats["limbs_loaded"]

        msg = f"Loaded {n_limbs} limb(s). {n_matched} bone(s) matched"
        if n_missing > 0:
            sample = sorted(match_stats["missing_bones"])[:5]
            sample_str = ", ".join(sample)
            if n_missing > 5:
                sample_str += f" (+{n_missing - 5} more)"
            msg += f". {n_missing} bone(s) missing on this armature: {sample_str}"
            if match_stats["limbs_with_missing_end_bone"]:
                msg += (f". Limbs with missing end bone: "
                        f"{', '.join(match_stats['limbs_with_missing_end_bone'])}")
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)

        return {"FINISHED"}


# ===========================================================================
# UI
# ===========================================================================

class DK_UL_limbs(UIList):
    """Limb list — shows name + capture status icons."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        captured_count = sum(1 for p in item.poses if p.captured)
        total = len(DIRECTIONS)
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon="BONE_DATA")
            row.label(text=f"{captured_count}/{total}",
                      icon=("CHECKMARK" if captured_count == total else "DOT"))
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


class DK_PT_panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Dummy Kin"
    bl_label = "Dummy Kinematics"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def draw(self, context):
        layout = self.layout
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin

        # --- Limb list ---
        box = layout.box()
        box.label(text="Limbs", icon="ARMATURE_DATA")
        row = box.row()
        row.template_list("DK_UL_limbs", "", dk, "limbs", dk, "active_limb_index",
                          rows=3)
        col = row.column(align=True)
        col.operator("dk.add_limb", icon="ADD", text="")
        col.operator("dk.remove_limb", icon="REMOVE", text="")

        if not (0 <= dk.active_limb_index < len(dk.limbs)):
            box.label(text="Add a limb to get started.", icon="INFO")
            return

        limb = dk.limbs[dk.active_limb_index]

        # --- Limb settings ---
        box = layout.box()
        box.label(text="Limb Setup", icon="BONE_DATA")

        # Display name — explicitly editable, distinct from the bone target.
        col = box.column(align=True)
        col.label(text="Name (what you call this limb):")
        col.prop(limb, "name", text="")

        col.separator()

        # End bone — the actual bone in the rig that this limb controls.
        col.label(text="Pick the ending bone (e.g. Foot, Hand, tip of tail, Tip of ear):")
        row = col.row(align=True)
        row.prop_search(limb, "end_bone", arm_obj.data, "bones", text="")
        row.operator("dk.set_end_bone_from_active", text="", icon="EYEDROPPER")

        # Rest position indicator — important because the blend math uses
        # rest position as the origin for direction calculations.
        rest_row = col.row(align=True)
        if limb.rest_position_valid:
            rest_row.label(
                text=(f"Rest: ({limb.rest_position[0]:.2f}, "
                      f"{limb.rest_position[1]:.2f}, "
                      f"{limb.rest_position[2]:.2f})"),
                icon="PINNED")
        else:
            rest_row.label(text="Rest: not set", icon="ERROR")
        rest_row.operator("dk.refresh_rest", text="", icon="FILE_REFRESH")

        if context.mode != "POSE":
            box.label(text="Enter Pose Mode to capture / restore.", icon="INFO")

        # --- Capture grid ---
        cap = layout.box()
        header = cap.row(align=True)
        header.label(text="Reference Captures", icon="REC")
        # Lock toggle on the header — single button that flips the state.
        lock_op = header.operator(
            "dk.toggle_lock",
            text="Locked" if limb.locked else "Unlocked",
            icon="LOCKED" if limb.locked else "UNLOCKED",
            depress=limb.locked,
        )

        # Compact mode hint so users know which mode is active
        mode_label = next((lbl for k, lbl, _ in CAPTURE_MODES
                           if k == dk.capture_mode), dk.capture_mode)
        cap.label(text=f"Mode: {mode_label}  (change in Settings)", icon="INFO")

        # The actual capture grid — entire section greys out when locked.
        grid_container = cap.column(align=True)
        grid_container.enabled = not limb.locked

        grid = grid_container.grid_flow(row_major=True, columns=2,
                                        even_columns=True, even_rows=False,
                                        align=True)
        for d_id, d_label, d_desc in DIRECTIONS:
            pose = limb.get_pose(d_id)
            captured = pose is not None and pose.captured
            sub = grid.row(align=True)
            op = sub.operator("dk.capture_direction",
                              text=f"{d_label}" + (" ✓" if captured else ""),
                              icon=("CHECKMARK" if captured else "REC"))
            op.direction = d_id
            clr = sub.operator("dk.clear_direction", text="", icon="X")
            clr.direction = d_id

        n_captured = sum(1 for p in limb.poses if p.captured)
        cap.label(text=f"{n_captured}/{len(DIRECTIONS)} directions captured",
                  icon=("CHECKMARK" if n_captured == len(DIRECTIONS) else "DOT"))

        # --- View captures readout (collapsed by default via expander) ---
        if n_captured > 0:
            view_box = cap.box()
            view_box.label(text="Captured offsets (relative to rest):",
                           icon="HIDE_OFF")
            origin = (Vector(limb.rest_position) if limb.rest_position_valid
                      else Vector((0, 0, 0)))
            for d_id, d_label, _ in DIRECTIONS:
                pose = limb.get_pose(d_id)
                if pose is None or not pose.captured:
                    continue
                rel = Vector(pose.end_offset) - origin
                n_bones = len(pose.bone_deltas)
                view_box.label(
                    text=f"  {d_label}: ({rel.x:+.2f}, {rel.y:+.2f}, "
                         f"{rel.z:+.2f})  [{n_bones} bone(s)]")

        # --- Save / Restore ---
        sr = layout.box()
        sr.label(text="Save & Restore", icon="FILE_TICK")

        row = sr.row(align=True)
        row.scale_y = 1.2
        row.operator("dk.save_pose", icon="FILE_TICK")
        row.operator("dk.restore_pose", icon="LOOP_BACK")

        if limb.saved_valid:
            sr.label(text=(f"Saved (world): ({limb.saved_target[0]:.3f}, "
                           f"{limb.saved_target[1]:.3f}, "
                           f"{limb.saved_target[2]:.3f})"),
                     icon="PINNED")
            sr.operator("dk.clear_saved", icon="X")
        else:
            sr.label(text="No target saved yet.", icon="INFO")

        # Version stamp at the very bottom of the main panel
        ver = bl_info["version"]
        foot = layout.row()
        foot.alignment = "RIGHT"
        foot.label(text=f"v{ver[0]}.{ver[1]}.{ver[2]}")


class DK_PT_presets(Panel):
    """Sub-panel for preset save/load. Lives under the main DK panel."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Dummy Kin"
    bl_label = "Presets"
    bl_parent_id = "DK_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def draw(self, context):
        layout = self.layout
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin

        col = layout.column(align=True)
        col.label(text="Save / Load DK Setup (.json)", icon="FILE_FOLDER")

        row = col.row(align=True)
        row.scale_y = 1.1
        save_op = row.operator("dk.save_preset", icon="EXPORT", text="Save Preset")
        load_op = row.operator("dk.load_preset", icon="IMPORT", text="Load Preset")

        col.separator()
        info = col.box()
        info.label(text=f"Limbs on this armature: {len(dk.limbs)}", icon="INFO")
        info.label(text="Loading appends by default.")
        info.label(text="Bone names must match the source rig.")
        info.label(text="Missing bones are skipped & reported.")


class DK_PT_settings(Panel):
    """Sub-panel for capture-mode + future preferences. Lives under main DK panel."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Dummy Kin"
    bl_label = "Settings"
    bl_parent_id = "DK_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def draw(self, context):
        layout = self.layout
        arm_obj = context.active_object
        dk = arm_obj.data.dummy_kin

        col = layout.column(align=True)
        col.label(text="Capture Mode", icon="REC")
        col.prop(dk, "capture_mode", text="")

        # Description for the current mode
        desc = next((d for k, _, d in CAPTURE_MODES if k == dk.capture_mode), "")
        if desc:
            box = col.box()
            # Word-wrap the description by splitting roughly
            words = desc.split()
            line = ""
            lines = []
            for w in words:
                if len(line) + len(w) + 1 > 50:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            for ln in lines:
                box.label(text=ln)


# ===========================================================================
# Registration
# ===========================================================================

classes = (
    DK_BoneDelta,
    DK_DirectionPose,
    DK_Limb,
    DK_ArmatureProps,
    DK_OT_add_limb,
    DK_OT_remove_limb,
    DK_OT_set_end_bone_from_active,
    DK_OT_refresh_rest,
    DK_OT_toggle_lock,
    DK_OT_capture_direction,
    DK_OT_clear_direction,
    DK_OT_save_pose,
    DK_OT_restore_pose,
    DK_OT_clear_saved,
    DK_OT_save_preset,
    DK_OT_load_preset,
    DK_UL_limbs,
    DK_PT_panel,
    DK_PT_presets,
    DK_PT_settings,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Per-armature data lives on the armature DATA block, not the object,
    # so duplicating the armature object doesn't fork the limb config.
    bpy.types.Armature.dummy_kin = PointerProperty(type=DK_ArmatureProps)


def unregister():
    if hasattr(bpy.types.Armature, "dummy_kin"):
        del bpy.types.Armature.dummy_kin
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
