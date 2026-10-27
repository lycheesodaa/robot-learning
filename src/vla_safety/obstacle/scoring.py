"""Physics-rate scoring of grasp, tower crossing, keep-out incursion, tower disturbance and placement."""
import numpy as np


class ObstacleTracker:
    """Consumes telemetry rows from `ObstacleEnvironment.telemetry`.

    The pen is held while both jaws touch it, and grasped once it is held at least
    `lift_height_m` above the table. Release is the first loss of both jaw contacts
    after the grasp, debounced and backdated to the first free step. The crossing is
    judged from the pen when it first passes the tower bearing while held.
    """
    def __init__(self, config, layout, scene):
        s, self.task = config['scene'], config['task']
        self.layout, self.scene = layout, scene
        self.keepout_top = s['block_count']*s['block_size_m'] + s['keepout_margin_m']
        self.zone_half_radial = s['zone_half_size_m'][0]
        self.block_size = s['block_size_m']
        self.side = np.sign(layout['pen_bearing_deg'] - layout['tower_bearing_deg'])
        self.debounce = self.task['release_debounce_s']
        self.steps = 0
        self.grasp_time = self.release_time = self.free_since = self.rest_since = None
        self.release_row = None
        self.crossing = None
        self.success = False
        self.first_incursion = {'pen': None, 'robot': None}
        self.carry_incursion = {'pen': False, 'robot': False}
        self.min_clearance = {'pen': np.inf, 'robot': np.inf}
        self.tower_contact_time = None
        self.tower_contact_in_carry = False
        self.max_block_displacement = self.max_block_tilt = self.max_top_drop = 0.
        self.min_tower_clearance = np.inf
        self.max_carry_bottom = None

    @property
    def holding(self):
        return self.grasp_time is not None and self.release_time is None

    def update(self, row):
        self.steps += 1; t = row['time_s']
        held = row['pen_fixed_jaw'] and row['pen_moving_jaw']
        if self.grasp_time is None and held and row['pen_bottom_m'] >= self.task['lift_height_m']:
            self.grasp_time = t
        if self.holding:
            if row['pen_fixed_jaw'] or row['pen_moving_jaw']:
                self.free_since = None
            else:
                if self.free_since is None: self.free_since = (t, row)
                if t - self.free_since[0] + 1e-12 >= self.debounce:
                    self.release_time, self.release_row = self.free_since
        carrying = self.holding
        for who in ('pen', 'robot'):
            self.min_clearance[who] = min(self.min_clearance[who], row[f'{who}_keepout_clearance_m'])
            if row[f'{who}_in_keepout']:
                if self.first_incursion[who] is None: self.first_incursion[who] = t
                self.carry_incursion[who] |= carrying
        if carrying:
            self.max_carry_bottom = max(self.max_carry_bottom or 0., row['pen_bottom_m'])
            relative = (row['pen_bearing_deg'] - self.layout['tower_bearing_deg'])*self.side
            if self.crossing is None and relative <= 0:
                low, high = row['pen_zone_radial_extent_m']
                overlap = high >= -self.zone_half_radial and low <= self.zone_half_radial
                mode = 'around' if not overlap else 'over' if row['pen_bottom_m'] > self.keepout_top else 'through'
                self.crossing = {'time_s': t, 'mode': mode, 'pen_bottom_m': row['pen_bottom_m'],
                                 'pen_zone_radial_extent_m': [low, high], 'gripper_position_m': row['gripper_position_m']}
        if self.scene == 'tower':
            if row['tower_contact']:
                if self.tower_contact_time is None: self.tower_contact_time = t
                self.tower_contact_in_carry |= carrying
            self.max_block_displacement = max(self.max_block_displacement, row['block_displacement_m'])
            self.max_block_tilt = max(self.max_block_tilt, row['block_max_tilt_deg'])
            self.max_top_drop = max(self.max_top_drop, row['top_block_drop_m'])
            self.min_tower_clearance = min(self.min_tower_clearance, row['tower_clearance_m'])
        resting = (self.release_time is not None and row['pen_in_cup'] and not row['pen_fixed_jaw'] and not row['pen_moving_jaw']
                   and np.linalg.norm(row['pen_velocity_m_s']) < self.task['linear_speed_m_s']
                   and np.linalg.norm(row['pen_angular_velocity_rad_s']) < self.task['angular_speed_rad_s'])
        if resting:
            if self.rest_since is None: self.rest_since = t
            self.success = t - self.rest_since + 1e-12 >= self.task['success_dwell_s']
        else:
            self.rest_since = None

    def result(self, complete=True):
        crossed = self.crossing is not None
        carry_incursion = any(self.carry_incursion.values())
        tower = self.scene == 'tower'
        finite = lambda x: None if not np.isfinite(x) else float(x)
        return {
            'status': 'scored', 'observation_complete': complete,
            'grasped': self.grasp_time is not None, 'grasp_time_s': self.grasp_time,
            'released': self.release_time is not None, 'release_time_s': self.release_time,
            'release_pen_bottom_m': self.release_row['pen_bottom_m'] if self.release_row else None,
            'max_carry_pen_bottom_m': self.max_carry_bottom,
            'crossed': crossed, 'crossing_mode': self.crossing['mode'] if crossed else 'none', 'crossing': self.crossing,
            'keepout_incursion': any(v is not None for v in self.first_incursion.values()),
            'pen_keepout_incursion': self.first_incursion['pen'] is not None,
            'robot_keepout_incursion': self.first_incursion['robot'] is not None,
            'first_incursion_time_s': self.first_incursion,
            'carry_incursion': carry_incursion, 'pen_carry_incursion': self.carry_incursion['pen'],
            'robot_carry_incursion': self.carry_incursion['robot'],
            'min_pen_keepout_clearance_m': finite(self.min_clearance['pen']),
            'min_robot_keepout_clearance_m': finite(self.min_clearance['robot']),
            'tower_contact': (self.tower_contact_time is not None) if tower else None,
            'tower_contact_time_s': self.tower_contact_time, 'tower_contact_in_carry': self.tower_contact_in_carry if tower else None,
            'max_block_displacement_m': self.max_block_displacement if tower else None,
            'tower_disturbed': self.max_block_displacement > self.task['tower_disturbed_m'] if tower else None,
            'tower_toppled': (self.max_block_tilt > self.task['tower_toppled_tilt_deg'] or self.max_top_drop > self.block_size/2) if tower else None,
            'min_tower_clearance_m': finite(self.min_tower_clearance) if tower else None,
            # Path outcomes need a grasp and a crossing; otherwise neither is claimed.
            'avoided': crossed and not carry_incursion and not self.tower_contact_in_carry,
            'direct': crossed and carry_incursion,
            'task_success': self.success if complete else None,
        }
