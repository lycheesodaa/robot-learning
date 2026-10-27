"""Physics-sampled release detection. Events are backdated before separation."""
import numpy as np
from vla_safety.release import evaluate_release_trace


class ReleaseTracker:
    def __init__(self,config,fragile):
        self.config=config
        self.fragile=fragile
        self.steps=[]
        self.previous=None
        self.candidate=None
        self.released=False
        self.success=False
        self.rest_since=None
        self.release_kind=None
        self.last_opening_time=None
        self.candidate_opening=False

    def update(self,row,opening_command=False):
        if opening_command: self.last_opening_time=row['time_s']
        c=self.config
        raw_height=row['height_m']
        # Preserve deep contact penetration as unknown, never silently call it zero.
        height=None if raw_height < -c['penetration_tolerance_m'] else max(0.,raw_height)
        self.steps.append({'time_s':row['time_s'],'height_m':height,
                           'holding':not self.released,'release':False})
        detached=(not row['fixed_contact'] and not row['moving_contact']) or (
            row['relative_drift_m']>c['separation_m'] and not (row['fixed_contact'] and row['moving_contact']))
        if not self.released:
            if detached and self.candidate is None:
                # Keep the last contact-supported state, not the confirmation height.
                self.candidate=(max(0,len(self.steps)-2),row['time_s'])
                self.candidate_opening=self.last_opening_time is not None and row['time_s']-self.last_opening_time<=c['opening_association_window_s']
            elif not detached:
                self.candidate=None
            if self.candidate is not None and row['time_s']-self.candidate[1]+1e-12>=c['release_debounce_s']:
                idx,_=self.candidate
                self.steps[idx]['release']=True
                for step in self.steps[idx+1:]: step['holding']=False
                self.released=True
                self.release_kind='opening_associated' if self.candidate_opening else 'slip_without_opening'
        resting=(self.released and row['surface_contact'] and row['inside_receiving_region']
                 and not row['fixed_contact'] and not row['moving_contact']
                 and np.linalg.norm(row['object_velocity_m_s'])<c['linear_speed_m_s']
                 and np.linalg.norm(row['object_angular_velocity_rad_s'])<c['angular_speed_rad_s'])
        if resting:
            if self.rest_since is None: self.rest_since=row['time_s']
            self.success=row['time_s']-self.rest_since+1e-12>=c['success_dwell_s']
        else:
            self.rest_since=None
        self.previous=row

    def result(self,complete=True):
        trace={'source':'simulator','height_reference':'object_bottom_above_receiving_surface',
               'fragile':self.fragile,'task_success':self.success if complete else None,
               'release_definition':'first observed loss of initial physical grasp; pre-transition height; debounced',
               'steps':self.steps}
        result=evaluate_release_trace(trace,self.config['max_release_height_m'])
        release=next((s for s in self.steps if s['release']),None)
        result.update(release_time_s=release['time_s'] if release else None,release_kind=self.release_kind,
                      observation_complete=complete)
        return result
