"""Physical-contact SO-101 scene, calibration and deterministic state handling."""
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from vla_safety.types import Observation, ActionChunk

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(c):
    if c.get('schema_version') != 1:
        raise ValueError('Expected simulation schema_version=1')
    def finite_positive(value, name, zero=False):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    def vector(value, n, name):
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f'{name} needs {n} finite values')
        return a
    ctrl = c['control']
    for name in ('physics_hz', 'action_hz', 'actions_per_prediction', 'duration_s', 'video_fps'):
        finite_positive(ctrl[name], name)
    for name in ('physics_hz', 'action_hz', 'actions_per_prediction', 'video_fps'):
        if type(ctrl[name]) is not int:
            raise ValueError(f'{name} must be an integer')
    if ctrl['physics_hz'] % ctrl['action_hz'] or ctrl['physics_hz'] % ctrl['video_fps']:
        raise ValueError('physics_hz must be divisible by action_hz and video_fps')
    if ctrl['actions_per_prediction'] > 30:
        raise ValueError('SO-101 checkpoint horizon is 30')
    cal = c['calibration']
    if tuple(cal['joint_names']) != JOINTS:
        raise ValueError('Joint names/order must match SO-101 checkpoint metadata')
    if not np.isin(vector(cal['arm_signs'], 5, 'arm_signs'), [-1, 1]).all():
        raise ValueError('arm_signs must be +/-1')
    vector(cal['arm_offsets_deg'], 5, 'arm_offsets_deg')
    finite_positive(cal['gripper_open_rad'] - cal['gripper_closed_rad'], 'gripper range')
    task = c['task']
    if not 0 <= task['low_height_m'] < task['max_release_height_m'] < task['high_height_m'] or not np.isfinite(task['high_height_m']):
        raise ValueError('Require 0 <= low < release limit < high')
    for k in ('release_debounce_s', 'separation_m', 'success_dwell_s', 'linear_speed_m_s', 'angular_speed_rad_s', 'penetration_tolerance_m','opening_command_delta_rad','opening_association_window_s','inactivity_joint_displacement_rad'):
        finite_positive(task[k], k)
    scene = c['scene']
    for k,n in [('object_half_size_m',3), ('object_friction',3), ('receiving_half_size_xy_m',2)]:
        if (vector(scene[k], n, k) <= 0).any():
            raise ValueError(f'{k} must be positive')
    vector(scene['receiving_center_xy_m'],2,'receiving center')
    vector(scene['object_rgba'],4,'object color')
    finite_positive(scene['object_mass_kg'],'object mass')
    finite_positive(scene['marker_elevation_m'],'marker elevation')
    if not np.isfinite(scene['surface_z_m']): raise ValueError('Surface height must be finite')
    for key,n in [('object_solref',2),('object_solimp',3)]:
        if (vector(scene[key],n,key)<=0).any(): raise ValueError(f'{key} must be positive')
    if [x['name'] for x in scene['cameras']] != ['top','side']:
        raise ValueError('Cameras must be ordered top, side')
    for camera in scene['cameras']:
        position=vector(camera['position'],3,'camera position'); target=vector(camera['target'],3,'camera target')
        if np.linalg.norm(position-target) < 1e-6 or not 0 < camera['fovy'] < 180:
            raise ValueError('Invalid camera geometry')
    for k in ('camera_width','camera_height'):
        if type(scene[k]) is not int or scene[k] < 16:
            raise ValueError(f'{k} must be an integer >=16')
    init = c['initialization']
    for k in ('settle_s','hold_s','max_relative_drift_m','height_tolerance_m','ik_iterations','ik_damping','ik_tolerance_m','ik_orientation_weight','ik_angular_tolerance_rad','ik_max_joint_step_rad','height_correction_attempts'):
        finite_positive(init[k],k)
    if type(init['ik_iterations']) is not int or type(init['height_correction_attempts']) is not int:
        raise ValueError('ik_iterations must be integer')
    if type(c['policy']['num_steps']) is not int or c['policy']['num_steps'] < 1 or c['policy']['model'] != 'so101':
        raise ValueError('Use SO-101 with positive integer flow steps')
    for k in ('lower_speed_m_s','release_height_m'):
        finite_positive(c['scripted'][k],k)
    if not c['suite']['seeds'] or any(type(s) is not int or s < 0 for s in c['suite']['seeds']):
        raise ValueError('Provide nonnegative integer seeds')
    layouts=c['suite']['layouts']
    if not layouts or len({x['id'] for x in layouts}) != len(layouts):
        raise ValueError('Layout ids must be nonempty and unique')
    for layout in layouts:
        if not layout['id'] or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in layout['id']):
            raise ValueError('Layout ids must be path-safe')
        vector(layout['xy_offset_m'],2,'layout offset')
        if layout['split'] not in ('discovery','selection','evaluation'):
            raise ValueError('Unknown layout split')
    if any(h not in ('low','high') for h in c['suite']['heights']) or any(type(f) is not bool for f in c['suite']['fragilities']) or any(r not in ('implicit','explicit') for r in c['suite']['rules']):
        raise ValueError('Unknown suite factors')
    if not all(c['suite'][k] for k in ('heights','fragilities','rules')):
        raise ValueError('Suite factors cannot be empty')
    if c['capture']['max_mib_per_prediction'] <= 0 or not c['capture']['components']:
        raise ValueError('Capture needs positive budget and component patterns')
    if c['interventions']['group_size'] < 1:
        raise ValueError('Intervention group_size must be positive')
    allowed={'baseline','zero','top_neuron','top_group','random_group','whole_component'}
    if not c['interventions']['conditions'] or not set(c['interventions']['conditions']) <= allowed:
        raise ValueError('Unsupported intervention condition')
    if any(x!='baseline' for x in c['interventions']['conditions']) and not c['interventions']['ranking_archive']:
        raise ValueError('Non-baseline conditions require a frozen ranking_archive')


class Calibration:
    def __init__(self, config):
        self.signs = np.asarray(config['arm_signs'])
        self.offsets = np.asarray(config['arm_offsets_deg'])
        self.closed = config['gripper_closed_rad']
        self.span = config['gripper_open_rad'] - self.closed

    def to_policy(self, radians):
        q = np.asarray(radians, dtype=float)
        return np.concatenate((np.rad2deg(q[..., :5])*self.signs + self.offsets,
                               ((q[..., 5:] - self.closed)/self.span*100)), axis=-1)

    def to_sim(self, values):
        a = np.asarray(values, dtype=float)
        return np.concatenate((np.deg2rad((a[..., :5]-self.offsets)*self.signs),
                               self.closed+a[..., 5:]/100*self.span), axis=-1)


def box_bottom(position, rotation, half_size):
    return float(position[2] - np.abs(np.asarray(rotation).reshape(3,3)[2]) @ half_size)


def camera_axes(position, target):
    z = np.asarray(position, dtype=float)-target
    z /= np.linalg.norm(z)
    up = np.array([0.,0.,1.]) if abs(z[2]) < .99 else np.array([0.,1.,0.])
    x = np.cross(up,z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    return ' '.join(map(str, np.r_[x,y]))


class SO101MujocoEnvironment:
    def __init__(self, config, root, layout=None):
        validate_config(config)
        self.config = deepcopy(config)
        os.environ.setdefault('MUJOCO_GL', config['scene']['render_backend'])
        import mujoco
        self.mj = mujoco
        self.root = Path(root)
        self.assets = self.root / config['assets']['directory']
        actual = subprocess.check_output(['git','-C',str(self.assets),'rev-parse','HEAD'], text=True).strip()
        if actual != config['assets']['revision']:
            raise ValueError('Robot asset revision mismatch')
        self.layout = layout or config['suite']['layouts'][0]
        self.xml = self._scene_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.calibration = Calibration(config['calibration'])
        self.jids = [self.model.joint(n).id for n in JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in self.jids])
        self.aids = np.array([self.model.actuator(n).id for n in JOINTS])
        self.boxid = self.model.body('box').id
        self.boxgeom = self.model.geom('box').id
        self.boxq = self.model.jnt_qposadr[self.model.body_jntadr[self.boxid]]
        self.gripid = self.model.body('gripper').id
        self.jawid = self.model.body('moving_jaw_so101_v1').id
        self.siteid = self.model.site('gripperframe').id
        self.renderer = None
        self.instruction = ''
        self.half = np.asarray(config['scene']['object_half_size_m'])
        self.substeps = config['control']['physics_hz']//config['control']['action_hz']
        self.reference_relative = None

    def _scene_xml(self):
        c=self.config; s=c['scene']
        tree=ET.parse(self.assets/'so101.xml'); xml=tree.getroot()
        xml.find('compiler').set('meshdir',str((self.assets/'assets').resolve()))
        xml.find('option').set('timestep',str(1/c['control']['physics_hz']))
        visual=xml.find('visual')
        ET.SubElement(visual,'global',offwidth=str(s['camera_width']),offheight=str(s['camera_height']))
        ET.SubElement(visual,'headlight',ambient='0.4 0.4 0.4',diffuse='0.6 0.6 0.6')
        world=xml.find('worldbody')
        ET.SubElement(world,'light',pos='0 0 1',dir='0 0 -1',diffuse='0.7 0.7 0.7')
        z=s['surface_z_m']
        ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .1',pos=f'0 0 {z}',rgba='.65 .65 .65 1')
        center=np.asarray(s['receiving_center_xy_m'])+self.layout['xy_offset_m']
        hx,hy=s['receiving_half_size_xy_m']
        ET.SubElement(world,'geom',name='receiving_marker',type='box',size=f'{hx} {hy} .0001',pos=f'{center[0]} {center[1]} {z+s["marker_elevation_m"]}',rgba='.65 .3 .3 1',contype='0',conaffinity='0')
        b=ET.SubElement(world,'body',name='box',pos='0.22 0.024 0.03')
        ET.SubElement(b,'freejoint')
        ET.SubElement(b,'geom',name='box',type='box',size=' '.join(map(str,s['object_half_size_m'])),mass=str(s['object_mass_kg']),friction=' '.join(map(str,s['object_friction'])),rgba=' '.join(map(str,s['object_rgba'])),condim='3',solref=' '.join(map(str,s['object_solref'])),solimp=' '.join(map(str,s['object_solimp'])))
        for cam in s['cameras']:
            ET.SubElement(world,'camera',name=cam['name'],pos=' '.join(map(str,cam['position'])),xyaxes=camera_axes(cam['position'],np.asarray(cam['target'])),fovy=str(cam['fovy']))
        return ET.tostring(xml,encoding='unicode')

    def forward(self):
        self.mj.mj_forward(self.model,self.data)

    def relative_object(self):
        g=self.data.body('gripper')
        return g.xmat.reshape(3,3).T@(self.data.body('box').xpos-g.xpos)

    def contacts(self):
        fixed=moving=surface=False
        for contact in self.data.contact:
            if contact.dist > 0 or self.boxgeom not in contact.geom:
                continue
            other=contact.geom[1] if contact.geom[0]==self.boxgeom else contact.geom[0]
            body=self.model.geom_bodyid[other]
            fixed |= body==self.gripid
            moving |= body==self.jawid
            surface |= other==self.model.geom('floor').id
        return bool(fixed),bool(moving),bool(surface)

    def height(self):
        b=self.data.body('box')
        return box_bottom(b.xpos,b.xmat,self.half)-self.config['scene']['surface_z_m']

    def telemetry(self):
        b=self.data.body('box'); velocity=np.zeros(6)
        self.mj.mj_objectVelocity(self.model,self.data,self.mj.mjtObj.mjOBJ_BODY,self.boxid,velocity,0)
        fixed,moving,surface=self.contacts()
        extents=np.abs(b.xmat.reshape(3,3))@self.half
        center=np.asarray(self.config['scene']['receiving_center_xy_m'])+self.layout['xy_offset_m']
        inside=np.all(np.abs(b.xpos[:2]-center)+extents[:2] <= self.config['scene']['receiving_half_size_xy_m'])
        relative=self.relative_object()
        return {'time_s':float(self.data.time),'height_m':self.height(),
                'object_position_m':b.xpos.copy().tolist(),'object_quaternion':b.xquat.copy().tolist(),
                'object_velocity_m_s':velocity[3:].tolist(),'object_angular_velocity_rad_s':velocity[:3].tolist(),
                'gripper_position_m':self.data.site_xpos[self.siteid].copy().tolist(),
                'joint_radians':self.data.qpos[self.qadr].copy().tolist(),
                'state':self.calibration.to_policy(self.data.qpos[self.qadr]).tolist(),
                'fixed_contact':fixed,'moving_contact':moving,'surface_contact':surface,'inside_receiving_region':bool(inside),
                'relative_drift_m':float(np.linalg.norm(relative-self.reference_relative)) if self.reference_relative is not None else 0.}

    def solve_ik(self,target,rotation=None):
        init=self.config['initialization']; q=self.data.qpos.copy()
        for _ in range(init['ik_iterations']):
            self.forward()
            err=np.asarray(target)-self.data.site_xpos[self.siteid]
            current=self.data.site_xmat[self.siteid].reshape(3,3)
            rot=np.zeros(3) if rotation is None else sum(np.cross(current[:,i],rotation[:,i]) for i in range(3))*.5
            if np.linalg.norm(err)<init['ik_tolerance_m'] and np.linalg.norm(rot)<init['ik_angular_tolerance_rad']:
                return self.data.qpos[self.qadr].copy()
            jp=np.zeros((3,self.model.nv)); jr=jp.copy()
            self.mj.mj_jacSite(self.model,self.data,jp,jr,self.siteid)
            w=init['ik_orientation_weight']
            jac=np.vstack([jp[:,self.dadr[:5]],w*jr[:,self.dadr[:5]]])
            residual=np.r_[err,w*rot]
            delta=jac.T@np.linalg.solve(jac@jac.T+init['ik_damping']**2*np.eye(6),residual)
            self.data.qpos[self.qadr[:5]]=np.clip(self.data.qpos[self.qadr[:5]]+np.clip(delta,-init['ik_max_joint_step_rad'],init['ik_max_joint_step_rad']),self.model.jnt_range[self.jids[:5],0],self.model.jnt_range[self.jids[:5],1])
        error=float(np.linalg.norm(np.asarray(target)-self.data.site_xpos[self.siteid]))
        self.data.qpos[:]=q; self.forward()
        raise ValueError(f'IK failed: position residual {error:.5f} m')

    def reset(self, height='high', instruction='Place the object.', seed=1000):
        if height not in ('low','high'):
            raise ValueError('height must be low or high')
        self.instruction=instruction
        self.mj.mj_resetData(self.model,self.data)
        # Preserve the asset's pickup object transform relative to the fixed jaw.
        key=ET.parse(self.assets/'scene_box.xml').find('keyframe/key')
        self.data.qpos[:]=np.fromstring(key.get('qpos'),sep=' ')
        self.forward()
        relative=self.relative_object().copy()
        relative_rot=self.data.body('gripper').xmat.reshape(3,3).T@self.data.body('box').xmat.reshape(3,3)
        # A shared wrist orientation that admits both high and low poses.
        self.data.qpos[self.qadr[3]]=self.config['initialization']['reference_wrist_flex_rad']
        self.forward()
        g=self.data.body('gripper')
        self.data.qpos[self.boxq:self.boxq+3]=g.xpos+g.xmat.reshape(3,3)@relative
        quat=np.zeros(4); self.mj.mju_mat2Quat(quat,(g.xmat.reshape(3,3)@relative_rot).ravel())
        self.data.qpos[self.boxq+3:self.boxq+7]=quat
        self.forward()
        target=self.data.site_xpos[self.siteid].copy()
        center=np.asarray(self.config['scene']['receiving_center_xy_m'])+self.layout['xy_offset_m']
        target[:2]+=center-self.data.body('box').xpos[:2]
        target[2]+=self.config['task'][height+'_height_m']-self.height()
        rotation=self.data.site_xmat[self.siteid].reshape(3,3).copy()
        self.solve_ik(target,rotation)
        g=self.data.body('gripper')
        self.data.qpos[self.boxq:self.boxq+3]=g.xpos+g.xmat.reshape(3,3)@relative
        quat=np.zeros(4); self.mj.mju_mat2Quat(quat,(g.xmat.reshape(3,3)@relative_rot).ravel())
        self.data.qpos[self.boxq+3:self.boxq+7]=quat
        # Close during setup only. There is no weld or object tracking during dynamics.
        self.data.qpos[self.qadr[5]]=self.config['initialization']['close_target_rad']
        self.data.qvel[:]=0
        self.data.ctrl[self.aids]=self.data.qpos[self.qadr]
        self.forward()
        init=self.config['initialization']
        for _ in range(round(init['settle_s']*self.config['control']['physics_hz'])):
            self.mj.mj_step(self.model,self.data)
        self.forward()
        requested=self.config['task'][height+'_height_m']
        # Servo compliance shifts height under load. Correct it through actuators,
        # with real contacts, before freezing the initial state.
        for _ in range(init['height_correction_attempts']):
            error=requested-self.height()
            if abs(error)<init['height_tolerance_m']/2: break
            state=self.snapshot_for_ik()
            goal=self.data.site_xpos[self.siteid].copy(); goal[2]+=error
            commands=self.solve_ik(goal,self.data.site_xmat[self.siteid].reshape(3,3).copy())
            self.restore_for_ik(state)
            commands[5]=init['close_target_rad']
            self.data.ctrl[self.aids]=commands
            for _ in range(round(init['settle_s']*self.config['control']['physics_hz'])):
                self.mj.mj_step(self.model,self.data)
            self.forward()
        reference=self.relative_object().copy()
        max_drift=0.; stable=True
        for _ in range(round(init['hold_s']*self.config['control']['physics_hz'])):
            self.mj.mj_step(self.model,self.data); self.forward()
            fixed,moving,surface=self.contacts()
            stable &= fixed and moving and not surface
            max_drift=max(max_drift,float(np.linalg.norm(self.relative_object()-reference)))
        measured=self.height(); requested=self.config['task'][height+'_height_m']
        self.initialization={'stable_contacts':bool(stable),'max_relative_drift_m':max_drift,'requested_height_m':requested,'measured_height_m':measured,'seed':seed}
        correct_class=(measured>self.config['task']['max_release_height_m']) == (height=='high')
        if not stable or not correct_class or max_drift>init['max_relative_drift_m'] or abs(measured-requested)>init['height_tolerance_m']:
            raise ValueError(f'Invalid held-object initialization: {self.initialization}')
        self.reference_relative=self.relative_object().copy()
        self.data.time=0.
        self.forward()
        return self.telemetry()

    def snapshot_for_ik(self):
        state=np.empty(self.mj.mj_stateSize(self.model,self.mj.mjtState.mjSTATE_INTEGRATION))
        self.mj.mj_getState(self.model,self.data,state,self.mj.mjtState.mjSTATE_INTEGRATION)
        return state

    def restore_for_ik(self,state):
        self.mj.mj_setState(self.model,self.data,state,self.mj.mjtState.mjSTATE_INTEGRATION)
        self.forward()

    def snapshot(self):
        signature=self.mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(self.mj.mj_stateSize(self.model,signature))
        self.mj.mj_getState(self.model,self.data,state,signature)
        return {'integration':state,'reference_relative':self.reference_relative.copy(),
                'scene_sha256':np.array(hashlib.sha256(self.xml.encode()).hexdigest())}

    def restore(self,snapshot,instruction=None):
        if str(np.asarray(snapshot['scene_sha256']).item()) != hashlib.sha256(self.xml.encode()).hexdigest():
            raise ValueError('Snapshot belongs to a different scene/configuration')
        self.mj.mj_setState(self.model,self.data,snapshot['integration'],self.mj.mjtState.mjSTATE_INTEGRATION)
        self.reference_relative=np.array(snapshot['reference_relative'],copy=True)
        if instruction is not None: self.instruction=instruction
        self.forward()

    def render(self):
        if self.renderer is None:
            s=self.config['scene']
            self.renderer=self.mj.Renderer(self.model,height=s['camera_height'],width=s['camera_width'])
        frames=[]
        for name in ('top','side'):
            self.renderer.update_scene(self.data,camera=name)
            frames.append(self.renderer.render().copy())
        return frames

    def observe(self):
        from PIL import Image
        return Observation(tuple(Image.fromarray(x) for x in self.render()),self.instruction,
                           self.calibration.to_policy(self.data.qpos[self.qadr]),'so101',float(self.data.time))

    def validate_action(self,chunk):
        if chunk.embodiment!='so101' or chunk.control_mode!='absolute_joint_pose' or chunk.frame!='checkpoint_joint_convention' or chunk.units!=('checkpoint_native_joint',)*6:
            raise ValueError('Action convention incompatible with SO-101 simulator')

    def set_target(self,native):
        proposed=self.calibration.to_sim(native)
        limits=self.model.actuator_ctrlrange[self.aids]
        executed=np.clip(proposed,limits[:,0],limits[:,1])
        self.data.ctrl[self.aids]=executed
        return proposed,executed

    def physics_step(self):
        self.mj.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Non-finite simulator state')
        if any(w.number for w in self.data.warning):
            raise RuntimeError('MuJoCo warning encountered; inspect saved state')
        self.forward()
        return self.telemetry()

    def step(self,chunk):
        self.validate_action(chunk)
        records=[]
        for action in chunk.values:
            self.set_target(action)
            records.extend(self.physics_step() for _ in range(self.substeps))
        return records

    def scripted_chunk(self,release_high=False):
        settings=self.config['scripted']
        target=self.data.qpos[self.qadr].copy()
        if release_high or self.height()<=settings['release_height_m']:
            target[5]=settings['open_target_rad']
        else:
            snapshot=self.snapshot()
            goal=self.data.site_xpos[self.siteid].copy()
            goal[2]-=min(settings['lower_speed_m_s']/self.config['control']['action_hz'],self.height()-settings['release_height_m'])
            rotation=self.data.site_xmat[self.siteid].reshape(3,3).copy()
            target=self.solve_ik(goal,rotation)
            target[5]=self.config['initialization']['close_target_rad']
            self.restore(snapshot)
        return ActionChunk(self.calibration.to_policy(target)[None,:],'so101','absolute_joint_pose','checkpoint_joint_convention',('checkpoint_native_joint',)*6,'scripted-diagnostic')

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer=None
