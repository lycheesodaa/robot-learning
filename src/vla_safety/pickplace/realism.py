"""Scene realism + camera-pose matching for the so101-nexus pick-and-place env.

Vendored unchanged apart from the texture lookup and the floor/lighting options of install() from ataghof/molmoact2-so101-sim
(Apache-2.0) as used in robot_learning_rulebased@facdfff; the v6 LoRA checkpoint was
trained on scenes built by this module, so keep it in step with that copy.

Matches the sim's look and viewpoints to the real SO-101 rigs MolmoAct2 was trained on:

  - CAMERA: swap the sim's nadir top-down overhead for the oblique third-person views the
    model card shows (a low near-horizontal front camera + a high over-shoulder camera,
    not top-down).
  - WHOLE-SCENE tuning: desaturate the physically-impossible pure colors, wood-texture the
    table, add a light wall background, real material response (specular/shininess/
    reflectance; MuJoCo's OpenGL renderer ignores metallic/roughness), a 3-light rig with
    one soft shadow caster, anti-aliasing, tuned headlight.
  - Arm recolor gold->orange + blue gripper to match the common orange-PLA SO-101 build.

Note from the experiments this repo documents: these changes did NOT move the success
rate (the gap was data, not pixels). They are kept because they cost nothing, make demos
and eval videos easier to read, and keep the sim visually closer to a real rig.

Usage:
    from molmoact2_so101_sim import realism
    realism.install()                      # patch colors + scene builder (before env build)
    env = PickAndPlaceEnv(config=..., render_mode="rgb_array")
    realism.tune_arm(env.unwrapped.model)  # post-compile material tuning
    realism.set_camera(env, "front_low")   # oblique front cam0

The wood texture ships inside the package (`molmoact2_so101_sim/assets/wood_table.png`) and
is loaded through package resources, so it resolves in a non-editable install. Point the
MOLMOACT2_SO101_TEXTURE_DIR environment variable at a directory holding a `wood_table.png`
to override it. The bundled texture is original procedural work (see
tools/generate_wood_texture.py) distributable under this repo's Apache-2.0 license.
"""
from __future__ import annotations

import os
import pathlib
import re

import mujoco
import numpy as np

import so101_nexus.constants as C
import so101_nexus.mujoco.pick_and_place as PP
import so101_nexus.object_slots as OS
import so101_nexus.scene as SCN


def _texture_path() -> str:
    """Absolute filesystem path to the wood-table texture (bundled package data by default;
    MOLMOACT2_SO101_TEXTURE_DIR overrides with a directory holding `wood_table.png`)."""
    override = os.environ.get("MOLMOACT2_SO101_TEXTURE_DIR")
    if override:
        return str(pathlib.Path(override) / "wood_table.png")
    return str(pathlib.Path(__file__).resolve().parent / "assets" / "wood_table.png")


TEX_PATH = _texture_path()

# Physically-plausible (desaturated, darker) albedos; pure 1/0 colors read as cartoon.
_REAL_COLORS = {
    "red":    [0.55, 0.13, 0.11, 1.0],
    "orange": [0.85, 0.40, 0.12, 1.0],
    "yellow": [0.78, 0.68, 0.16, 1.0],
    "green":  [0.18, 0.44, 0.20, 1.0],
    "blue":   [0.14, 0.26, 0.52, 1.0],
    "purple": [0.40, 0.18, 0.45, 1.0],
    "black":  [0.08, 0.08, 0.09, 1.0],
    "white":  [0.90, 0.90, 0.88, 1.0],
    "gray":   [0.58, 0.57, 0.55, 1.0],
}

# Realism asset block (textures + materials) injected into the scene XML.
_ASSETS = f"""  <statistic extent="0.8" center="0.18 0 0.12"/>
  <asset>
    <texture name="rk_sky" type="skybox" builtin="gradient" rgb1="0.82 0.82 0.84" rgb2="0.52 0.53 0.55" width="512" height="3072"/>
    <texture name="rk_wood" type="2d" file="{TEX_PATH}"/>
    <material name="rk_table" texture="rk_wood" texuniform="true" texrepeat="3 3" specular="0.2" shininess="0.2" reflectance="0.03"/>
    <material name="rk_plain" rgba="0.80 0.78 0.74 1" specular="0.05" shininess="0.05" reflectance="0.0"/>
    <material name="rk_wall" rgba="0.87 0.86 0.83 1" specular="0.03" shininess="0.02" reflectance="0.0"/>
    <material name="rk_cube" specular="0.35" shininess="0.4" reflectance="0.02"/>
    <material name="rk_disc" specular="0.25" shininess="0.25" reflectance="0.0"/>
    <material name="rk_glass" specular="0.9" shininess="0.9" reflectance="0.15"/>
  </asset>
"""

# Tuned <visual>: kill the flat camera fill, AA on, crisp shadows, light haze/room.
_VISUAL = """  <visual>
    <headlight ambient="0.32 0.32 0.33" diffuse="0.22 0.22 0.22" specular="0 0 0"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="1.1" shadowscale="0.65" fogstart="4" fogend="14" haze="0.10"/>
    <rgba haze="0.80 0.80 0.82 1"/>
  </visual>"""

# 3-light studio rig: one warm spot key (soft shadow) + cool fill + rim (no shadow).
_LIGHTS = """    <light name="rk_key" type="spot" pos="0.55 0.45 1.35" dir="-0.35 -0.30 -1"
           diffuse="0.95 0.92 0.85" specular="0.20 0.20 0.20" cutoff="62" exponent="12" castshadow="true"/>
    <light name="rk_fill" type="directional" dir="0.5 0.25 -1" diffuse="0.34 0.36 0.40"
           specular="0 0 0" castshadow="false"/>
    <light name="rk_rim" type="directional" dir="-0.4 -0.7 -0.5" diffuse="0.18 0.18 0.22"
           specular="0.12 0.12 0.12" castshadow="false"/>"""

# Indoor rig, not in rulebased: two soft ceiling lights (one shadow caster) + a cool window fill,
# for the flat, slightly warm room lighting of the community SO-100/101 datasets.
_VISUAL_INDOOR = """  <visual>
    <headlight ambient="0.28 0.28 0.28" diffuse="0.08 0.08 0.08" specular="0 0 0"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="1.1" shadowscale="0.5" haze="0.05"/>
  </visual>"""
_LIGHTS_INDOOR = """    <light name="rk_ceiling1" type="spot" pos="0.30 0.25 1.4" dir="-0.1 -0.15 -1"
           diffuse="0.50 0.48 0.44" specular="0.05 0.05 0.05" cutoff="60" exponent="2" castshadow="true"/>
    <light name="rk_ceiling2" type="spot" pos="0.10 -0.45 1.6" dir="0.1 0.3 -1"
           diffuse="0.30 0.29 0.27" specular="0 0 0" cutoff="70" exponent="2" castshadow="false"/>
    <light name="rk_window" type="directional" dir="0.3 0.8 -0.5" diffuse="0.16 0.17 0.19"
           specular="0 0 0" castshadow="false"/>"""
_RIGS = {"realism": (_VISUAL, _LIGHTS), "indoor": (_VISUAL_INDOOR, _LIGHTS_INDOOR)}

# A light back wall so the background is a wall, not a void (behind the +X workspace).
_WALL = ('    <geom name="rk_wall" type="box" pos="0.75 0 0.45" size="0.02 1.2 0.6" '
         'material="rk_wall" contype="0" conaffinity="0"/>\n')
# backdrop 'room', not in rulebased: a second wall behind the arm, the background of a camera facing it.
_ROOM_WALL = ('    <geom name="rk_wall_back" type="box" pos="-0.40 0 0.45" size="0.02 1.2 0.6" '
              'material="rk_wall" contype="0" conaffinity="0"/>\n')

# Generalization variants, not in rulebased: a clear glass box in place of the red cube (same geometry and
# physics, only its look), and a blue bowl on the goal (the disc is hidden; see PickPlaceEnvironment.reset).
_GLASS_RGBA = "0.80 0.90 0.92 0.35"
_BOWL = {"base_radius": 0.035, "rim_radius": 0.055, "height": 0.03, "wall": 0.003, "segments": 48, "colliders": 24}


def _bowl_xml():
    """Visual lathed bowl mesh + hidden collision base and tilted wall boxes, in the target body's frame."""
    b = _BOWL; rb, R, H, t, n = b["base_radius"], b["rim_radius"], b["height"], b["wall"], b["segments"]
    # (r, z) cross-section, outside bottom centre -> outer wall -> rim -> inner wall -> inside floor centre.
    profile = [(0., 0.), (rb, 0.), (R + t/2, H), (R - t/2, H), (rb - t/2, t), (0., t)]
    verts, ring = [(0., 0., 0.)], {}
    for i, (r, z) in enumerate(profile[1:-1], start=1):
        ring[i] = len(verts)
        verts += [(r*np.cos(2*np.pi*k/n), r*np.sin(2*np.pi*k/n), z) for k in range(n)]
    top = len(verts); verts.append((0., 0., t))
    faces = []
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((0, ring[1] + k2, ring[1] + k))
        for i in range(1, len(profile) - 2):
            a, c = ring[i], ring[i + 1]
            faces += [(a + k, a + k2, c + k2), (a + k, c + k2, c + k)]
        last = ring[len(profile) - 2]
        faces.append((top, last + k, last + k2))
    vertex = " ".join(f"{v:.5f}" for p in verts for v in p)
    face = " ".join(str(i) for f in faces for i in f)
    slope = np.arctan2(R - rb, H); half_len = np.hypot(R - rb, H)/2; m = b["colliders"]
    half_width = np.pi*(R + t)/m*1.1
    walls = ""
    for k in range(m):
        theta = 2*np.pi*k/m
        qz = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)]); qy = np.array([np.cos(slope/2), 0, np.sin(slope/2), 0])
        q = np.zeros(4); mujoco.mju_mulQuat(q, qz, qy)
        rc = (rb + R)/2
        walls += (f'      <geom name="rk_bowl_wall{k}" type="box" size="{t/2} {half_width:.5f} {half_len:.5f}" '
                  f'pos="{rc*np.cos(theta):.5f} {rc*np.sin(theta):.5f} {H/2}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
                  'group="3" contype="1" conaffinity="1" condim="4"/>\n')
    asset = f'    <mesh name="rk_bowl" vertex="{vertex}" face="{face}" inertia="shell"/>\n'
    rgba = " ".join(str(x) for x in C.COLOR_MAP["blue"])
    body = (f'      <geom name="rk_bowl_visual" type="mesh" mesh="rk_bowl" rgba="{rgba}" material="rk_disc" '
            'contype="0" conaffinity="0" group="1"/>\n'
            f'      <geom name="rk_bowl_base" type="cylinder" size="{rb} {t/2}" pos="0 0 {t/2}" '
            'group="3" contype="1" conaffinity="1" condim="4"/>\n' + walls)
    return asset, body


_installed = {"done": False}
# install() options: floor 'wood' (rulebased) or 'plain_light'; lighting 'realism' (rulebased rig), 'indoor' or
# 'default' (so101-nexus); carried object 'cube' (rulebased) or 'glass_box'; target 'disc' (rulebased) or 'bowl';
# backdrop 'wall' (rulebased) or 'room'.
_options = {"floor": "wood", "lighting": "realism", "object": "cube", "target": "disc", "backdrop": "wall"}
if not hasattr(OS, "_rk_orig_build"):
    OS._rk_orig_build = OS.build_object_scene_xml


def _realistic_build(*args, **kwargs):
    xml = OS._rk_orig_build(*args, **kwargs)
    visual, lights = _RIGS.get(_options["lighting"], (SCN.SCENE_VISUAL_XML, SCN.SCENE_LIGHTS_XML))
    xml = xml.replace(SCN.SCENE_VISUAL_XML, visual, 1)
    walls = _WALL + (_ROOM_WALL if _options["backdrop"] == "room" else "")
    xml = xml.replace(SCN.SCENE_LIGHTS_XML, lights + "\n" + walls, 1)
    xml = xml.replace('<compiler angle="radian"/>', '<compiler angle="radian"/>\n' + _ASSETS, 1)
    # floor plane -> wood table (or plain light) material (drop the flat rgba)
    floor = "rk_table" if _options["floor"] == "wood" else "rk_plain"
    xml = re.sub(r'(<geom name="floor" type="plane"[^>]*?)\s+rgba="[^"]*"', rf'\1 material="{floor}"', xml)
    # cube geom -> add specular material (keeps its desaturated rgba for color), or clear glass
    if _options["object"] == "glass_box":
        xml = re.sub(r'(<geom name="\w+_geom" type="box"[^>]*?)\s+rgba="[^"]*"', rf'\1 rgba="{_GLASS_RGBA}" material="rk_glass"', xml)
    else:
        xml = re.sub(r'(<geom name="\w+_geom" type="box")', r'\1 material="rk_cube"', xml)
    # target disc -> add specular material
    xml = re.sub(r'(<geom name="target_disc" type="cylinder")', r'\1 material="rk_disc"', xml)
    if _options["target"] == "bowl":
        asset, body = _bowl_xml()
        xml = xml.replace("  </asset>\n", asset + "  </asset>\n", 1)
        xml = re.sub(r'(<body name="target"[^>]*>\n)', lambda m: m.group(1) + body, xml, count=1)
    return xml


def install(floor="wood", lighting="realism", obj="cube", target="disc", backdrop="wall"):
    """Desaturate COLOR_MAP (in place) + monkeypatch the scene builder. Call BEFORE building the env."""
    if floor not in ("wood", "plain_light") or lighting not in ("realism", "indoor", "default"):
        raise ValueError(f"Unknown floor={floor!r} or lighting={lighting!r}")
    if obj not in ("cube", "glass_box") or target not in ("disc", "bowl") or backdrop not in ("wall", "room"):
        raise ValueError(f"Unknown object={obj!r}, target={target!r} or backdrop={backdrop!r}")
    _options.update(floor=floor, lighting=lighting, object=obj, target=target, backdrop=backdrop)
    if _installed["done"]:
        return
    for k, v in _REAL_COLORS.items():
        if k in C.COLOR_MAP:
            C.COLOR_MAP[k] = list(v)  # mutate shared dict -> all importers see it
    OS.build_object_scene_xml = _realistic_build
    PP.build_object_scene_xml = _realistic_build  # pick_and_place imported it by name
    _installed["done"] = True


def tune_arm(model):
    """Post-compile material tuning. MuJoCo's native renderer honors specular/shininess/
    reflectance/rgba only. Recolor gold->orange, gripper->blue, give metal servos a sheen."""
    orange = np.array([0.88, 0.42, 0.12, 1.0])
    blue = np.array([0.15, 0.30, 0.60, 1.0])
    for i in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        rgba = model.mat_rgba[i].copy()
        is_gold = abs(rgba[0] - 1.0) < 0.05 and abs(rgba[1] - 0.82) < 0.05 and abs(rgba[2] - 0.12) < 0.05
        is_black = rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15
        if "jaw" in name or "gripper" in name or "follower" in name:
            model.mat_rgba[i] = blue
            model.mat_specular[i] = 0.5
            model.mat_shininess[i] = 0.4
        elif is_gold:
            model.mat_rgba[i] = orange
            model.mat_specular[i] = 0.45
            model.mat_shininess[i] = 0.35
            model.mat_reflectance[i] = 0.05
        elif is_black:
            model.mat_specular[i] = 0.5
            model.mat_shininess[i] = 0.5
            model.mat_reflectance[i] = 0.1


# Camera presets: eye + look-at (metres, base frame +X forward), matched to the real
# SO-101 rigs documented for MolmoAct2 (RealSense front + over-shoulder placements).
_CAM_PRESETS = {
    # low, near-horizontal front-corner view (~13 deg down)
    "front_low": (np.array([0.50, -0.25, 0.14]), np.array([0.20, 0.05, 0.04]), 58.0),
    # 40cm high, 27cm from the arm, 45 deg down, ~78 deg HFoV
    "front_45": (np.array([0.60, -0.10, 0.42]), np.array([0.22, 0.0, 0.03]), 60.0),
    # high over-the-shoulder (~48 deg down)
    "shoulder": (np.array([-0.10, -0.05, 0.45]), np.array([0.28, 0.0, 0.02]), 58.0),
    # near-overhead (~70 deg down), base at the top of the image (as the overhead camera of
    # Beegbrain/pick_lemon_and_drop_in_bowl); not in rulebased. set_camera renders from the
    # look-at's far side of `eye` in x, so eye.x < target.x puts the camera beyond the workspace.
    "top_down": (np.array([0.02, 0.0, 0.58]), np.array([0.22, 0.0, 0.0]), 58.0),
    # camera opposite the arm, facing it ~38 deg down from x=0.45, z=0.28 (true position; mirrored here as
    # above): the 'laptop'/'front' webcam of most MolmoAct2 SO-100/101 datasets. Not in rulebased.
    "front_facing": (np.array([-0.19, 0.0, 0.28]), np.array([0.13, 0.0, 0.03]), 55.0),
}


def set_camera(env, preset="front_low"):
    """Point the sim's overhead observation camera at one of the matched real-rig views."""
    U = env.unwrapped
    eye, target, fovy = _CAM_PRESETS[preset]
    d = eye - target
    dist = float(np.linalg.norm(d))
    cam = U._overhead_obs_cam
    cam.lookat[:] = target
    cam.distance = dist
    cam.elevation = float(-np.degrees(np.arcsin(d[2] / dist)))   # negative = looking down
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    U.model.vis.global_.fovy = float(fovy)
    return preset
