"""Data viewer for the PRISM dataset.

Loads a multimodal motion-capture sequence (SMPL pose, IMU, insole pressure,
SLAM head trajectory and environment, and scene objects) from a pickle file
and visualizes it interactively with aitviewer.
"""

import argparse
import os
import pickle
import random

import numpy as np
from scipy.signal import butter, filtfilt

from aitviewer.configuration import CONFIG as C
C._conf.z_up = True

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.scene.node import Node
from aitviewer.viewer import Viewer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from body part name to the index used in the IMU lookup tables.
BODY_IDX = {
    'L_Foot':  0, 'R_Foot':  1,
    'L_Wrist': 2, 'R_Wrist': 3,
    'Head':    4, 'Pelvis':  5,
    'L_Knee':  6, 'R_Knee':  7,
}

# SMPL vertex / joint indices that approximate where each IMU is mounted.
IMU_VERTEX_IDX = [3438, 6838, 2208, 5669, 410, 3021, 1176, 4663]
IMU_JOINT_IDX  = [10, 11, 20, 21, 15, 0, 4, 5]

# SMPL vertex IDs for the four insole corners (L/R x Front/Back).
FOOT_CONTACT_VERTEX_IDX = {'LF': 3222, 'LB': 3386, 'RF': 6620, 'RB': 6787}

DEFAULT_BODY_COLOR = (149 / 255, 149 / 255, 149 / 255, 1.0)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def butterworth_filter(data, cutoff=7, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def calculate_velocity(positions, fps=100):
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) * fps / 2
    velocity[0]    = (positions[1] - positions[0]) * fps
    velocity[-1]   = (positions[-1] - positions[-2]) * fps
    return velocity


def calculate_acceleration(velocity, fps=100):
    acceleration = np.zeros_like(velocity)
    acceleration[1:-1] = (velocity[2:] - velocity[:-2]) * fps / 2
    acceleration[0]    = (velocity[1] - velocity[0]) * fps
    acceleration[-1]   = (velocity[-1] - velocity[-2]) * fps
    return acceleration


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def set_smpl_sequence(pose, trans, ori, beta, gender,
                      color=DEFAULT_BODY_COLOR, z_up=False):
    smpl_layer = SMPLLayer(model_type='smpl', gender=gender, device=C.device)
    return SMPLSequence(
        poses_body=pose,
        poses_root=ori,
        betas=beta,
        trans=trans,
        is_rigged=True,
        smpl_layer=smpl_layer,
        color=color,
        z_up=z_up,
    )


def build_mocap_data(smpl_sequence):
    """Derive per-IMU position / velocity / acceleration / orientation from SMPL."""
    mocap_data = {'smpl_sequence': smpl_sequence}
    for key, idx in BODY_IDX.items():
        if not IMU_VERTEX_IDX[idx]:
            continue
        pos = smpl_sequence.vertices[:, IMU_VERTEX_IDX[idx]]
        vel = calculate_velocity(butterworth_filter(pos))
        acc = calculate_acceleration(butterworth_filter(vel))
        ori = smpl_sequence.rbs.rb_ori[:, IMU_JOINT_IDX[idx]]
        mocap_data[key] = {'pos': pos, 'vel': vel, 'acc': acc, 'ori': ori}
    return mocap_data


# ---------------------------------------------------------------------------
# Scene visualization helpers
# ---------------------------------------------------------------------------

def vis_imu_rigid_body(viewer, mocap_data, sensor_data, vis_flags,
                       color=(1.0, 0.0, 1.0, 1.0)):
    imu_rbs = Node(name='IMU Orientation')
    for key, flag in vis_flags.items():
        if not flag:
            continue
        pos = np.expand_dims(mocap_data[key]['pos'], axis=1)
        ori = np.expand_dims(sensor_data[key]['ori_world'], axis=1)
        if pos.shape[0] != ori.shape[0]:
            print('Error: %s' % key)
            print('pos shape:', pos.shape)
            print('ori shape:', ori.shape)
            continue
        rb = RigidBodies(pos, ori, length=0.1, gui_affine=False, name=key,
                         color=color, radius=0.001, radius_cylinder=0.005)
        imu_rbs.add(rb)
    viewer.scene.add(imu_rbs)


def vis_point_clouds(viewer, mocap_data, slam_data, vis_flags):
    traj_pcs = Node(name='Trajectory')

    if vis_flags['Mocap_L_Foot']:
        traj = mocap_data['L_Foot']['pos'].reshape(1, -1, 3)
        traj_pcs.add(PointClouds(traj, color=(102 / 255, 153 / 255, 255 / 255, 0.7),
                                 point_size=5.0, name='Mocap Left Foot'))

    if vis_flags['Mocap_R_Foot']:
        traj = mocap_data['R_Foot']['pos'].reshape(1, -1, 3)
        traj_pcs.add(PointClouds(traj, color=(102 / 255, 153 / 255, 255 / 255, 0.7),
                                 point_size=5.0, name='Mocap Right Foot'))

    if vis_flags['Aria_Head'] and slam_data is not None:
        traj = slam_data['head_traj'].reshape(1, -1, 3)
        traj_pcs.add(PointClouds(traj, color=(204 / 255, 153 / 255, 255 / 255, 0.7),
                                 point_size=5.0, name='Aria Head'))

    if vis_flags['Mocap_Head']:
        traj = mocap_data['Head']['pos'].reshape(1, -1, 3)
        traj_pcs.add(PointClouds(traj, color=(102 / 255, 204 / 255, 153 / 255, 0.7),
                                 point_size=5.0, name='Mocap Head'))

    viewer.scene.add(traj_pcs)

    if vis_flags['Env_Point_Clouds'] and slam_data is not None:
        points = slam_data['points'].reshape(1, -1, 3)
        viewer.scene.add(PointClouds(points, color=(0.5, 0.5, 0.5, 0.7),
                                     point_size=1.0, name='Point Cloud'))


def body_tracking_camera(viewer, mocap_data):
    """Smooth orbital camera that keeps the SMPL body centered."""
    targets = mocap_data['smpl_sequence'].vertices[:, 0].copy()
    targets[:, 2] = 1
    targets = butterworth_filter(targets, cutoff=0.5)

    center = (0, 0, 0.5)
    radius = 4
    num = 2000
    start_angle, end_angle = 90, 450

    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num=num)
    c = np.column_stack((np.cos(angles) * radius,
                         np.sin(angles) * radius,
                         np.zeros(angles.shape)))
    circle = c + center

    num_circle = targets.shape[0] // 1000 + 1
    repeated_circle = np.tile(circle, (num_circle, 1))[:targets.shape[0]]
    positions = targets + repeated_circle

    cam = PinholeCamera(positions, targets,
                        viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    cam.name = 'Body Tracking Camera'
    viewer.scene.add(cam)
    viewer.set_temp_camera(cam)


def foot_tracking_camera(viewer, mocap_data):
    """Static-offset camera that follows the left foot joint."""
    targets = mocap_data['smpl_sequence'].joints[:, 10].copy()
    targets = butterworth_filter(targets, 0.1).copy()
    positions = targets + np.array([2, 2, 0.5])

    cam = PinholeCamera(positions, targets,
                        viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    cam.name = 'Foot Tracking Camera'
    viewer.scene.add(cam)


def vis_contact(viewer, mocap_data, sensor_data):
    """Color-code the four insole corners by their on/off contact state."""
    contacts_left  = sensor_data['L_Foot']['contacts']   # [F, 2]: front, back
    contacts_right = sensor_data['R_Foot']['contacts']   # [F, 2]: front, back
    F = contacts_left.shape[0]

    spec = [
        ('Left Front',  'LF', contacts_left[:, 0]),
        ('Left Back',   'LB', contacts_left[:, 1]),
        ('Right Front', 'RF', contacts_right[:, 0]),
        ('Right Back',  'RB', contacts_right[:, 1]),
    ]

    contact_pcs = Node(name='Foot Contact')
    for name, vertex_key, on in spec:
        pos = mocap_data['smpl_sequence'].vertices[:, FOOT_CONTACT_VERTEX_IDX[vertex_key]]
        colors = np.array([(1.0, 0.0, 0.0, 1.0)] * F)
        colors[on == 1] = (0.0, 1.0, 0.0, 1.0)
        contact_pcs.add(PointClouds(pos[:, None, :], color=(1.0, 0.0, 0.0, 1.0),
                                    point_size=20.0, name=name,
                                    colors=colors[:, None, :]))
    viewer.scene.add(contact_pcs)


def visualize_arrow(vectors, origins, name, color=(1, 0, 0, 1), magnitude=1):
    vectors = vectors.copy() * magnitude
    tips = origins + vectors
    return Arrows(
        origins=origins.reshape(-1, 1, 3),
        tips=tips.reshape(-1, 1, 3),
        r_base=0.01, r_head=0.02, p=0.25,
        color=color, name=name,
    )


def vis_acc_vel_arrows(viewer, mocap_data, sensor_data, vis_flags, synth_imu_frames=None):
    arrows_all = Node(name='IMU Acceleration')
    for key in mocap_data.keys():
        if key == 'smpl_sequence' or vis_flags[key] is False:
            continue
        arrows = Node(name=key)

        acc = mocap_data[key]['acc']
        vel = mocap_data[key]['vel']
        pos = mocap_data[key]['pos']
        acc_imu = sensor_data[key]['acc_world_filt']

        if synth_imu_frames is not None and key in synth_imu_frames:
            acc_syn = np.zeros_like(acc)
            acc_syn[synth_imu_frames[key]] = acc[synth_imu_frames[key]]
            acc[synth_imu_frames[key]]     = np.zeros(3)
            acc_imu[synth_imu_frames[key]] = np.zeros(3)
            acc_arrows_imu_syn = visualize_arrow(acc_syn, pos, name='acc_imu_syn',
                                                 color=(0, 1, 1, 1), magnitude=0.1)
            arrows.add(acc_arrows_imu_syn)

        # acc_arrows_gt     = visualize_arrow(acc,     pos, name='acc_gt',     color=(1, 0, 0, 1), magnitude=0.1)
        # vel_arrows_gt     = visualize_arrow(vel,     pos, name='vel_gt',     color=(0, 0, 1, 1), magnitude=0.5)
        # arrows.add(acc_arrows, vel_arrows_gt)

        acc_arrows = visualize_arrow(acc_imu, pos, name='acc', color=(0, 1, 0, 1), magnitude=0.1)
        arrows.add(acc_arrows)

        arrows_all.add(arrows)
    viewer.scene.add(arrows_all)


def vis_vforce(viewer, insole_data):
    """Draw vertical ground reaction force vectors at the center of pressure (world frame)."""
    node = Node(name='Ground Reaction Force')
    for side in ['L_Foot', 'R_Foot', 'combined']:
        vForce_vec = insole_data[side]['force_world']
        cop_world  = insole_data[side]['CoP_world']
        color = (255 / 255, 140 / 255, 0 / 255, 1) if side == 'combined' \
           else (255 / 255, 215 / 255, 0 / 255, 1)
        node.add(visualize_arrow(vForce_vec, cop_world,
                                 name=f'vForce_{side}', color=color, magnitude=0.05))
    viewer.scene.add(node)


# ---------------------------------------------------------------------------
# Per-sequence orchestration
# ---------------------------------------------------------------------------

def visualize_sequence(data):
    """Build a viewer scene for one PRISM sequence and run it (blocking)."""
    data_info   = data['info']['data_info']
    insole_data = data['insole']
    imu_data    = data['imu']
    slam_data   = data['slam']
    objects     = data['objects']

    smpl_params = data['smpl_params']
    pose   = smpl_params['poses'][:, 3:]
    ori    = smpl_params['poses'][:, :3]
    trans  = smpl_params['trans'] + smpl_params['root_offset']
    beta   = smpl_params['betas']
    gender = smpl_params['gender']

    smpl_sequence = set_smpl_sequence(pose, trans, ori, beta, gender)
    mocap_data = build_mocap_data(smpl_sequence)

    obj_meshes = None
    if objects is not None:
        obj_meshes = {}
        for obj_name, obj in objects.items():
            face_colors = np.ones((1, obj['faces'].shape[0], 4)) * np.array([0.5, 0.5, 0.5, 1.0])
            obj_meshes[obj_name] = Meshes(
                vertices=obj['vertices'], faces=obj['faces'],
                name=obj_name, face_colors=face_colors,
            )

    # Per-element visibility toggles.
    vis_imu_rbs = dict.fromkeys(BODY_IDX, True)
    vis_imu_arr = dict.fromkeys(BODY_IDX, True)
    vis_pc_traj = {
        'Mocap_L_Foot': True, 'Mocap_R_Foot': True,
        'OpenGo_L_Foot': True, 'OpenGo_R_Foot': True,
        'Aria_Head': True, 'Mocap_Head': True,
        'Env_Point_Clouds': True,
    }

    viewer = Viewer()

    viewer.scene.add(smpl_sequence)
    vis_imu_rigid_body(viewer, mocap_data, imu_data, vis_imu_rbs)
    vis_point_clouds(viewer, mocap_data, slam_data, vis_pc_traj)
    body_tracking_camera(viewer, mocap_data)
    # foot_tracking_camera(viewer, mocap_data)
    vis_contact(viewer, mocap_data, insole_data)
    vis_acc_vel_arrows(viewer, mocap_data, imu_data, vis_imu_arr,
                       data_info['synth_imu_frames'])
    vis_vforce(viewer, insole_data)

    if obj_meshes is not None:
        object_rbs = Node(name='Object')
        for obj_mesh in obj_meshes.values():
            object_rbs.add(obj_mesh)
        viewer.scene.add(object_rbs)

    # Viewer settings.
    viewer.auto_set_floor = False
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = True
    viewer.scene.fps = 30.0
    viewer.playback_fps = 100.0
    viewer.shadows_enabled = True
    viewer.auto_set_camera_target = False
    viewer.run()


def iter_sequence_paths(data_dir, subject=None, take=None):
    """Yield (subj, take, path) for every `<data_dir>/<subj>/<take>.pkl`,
    optionally filtered by subject and/or take name (e.g. 'subj001', 'take002').
    Empty pkl files are skipped silently.
    """
    for subj in sorted(os.listdir(data_dir)):
        if subject is not None and subj != subject:
            continue
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.pkl'):
                continue
            t = os.path.splitext(fname)[0]
            if take is not None and t != take:
                continue
            path = os.path.join(subj_dir, fname)
            if os.path.getsize(path) == 0:
                continue
            yield subj, t, path


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize PRISM motion-capture sequences with aitviewer. '
                    'Without filters, plays all sequences in random order with looping.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-dir', default='data/PRISM',
                   help='Root directory containing <subject>/<take>.pkl files.')
    p.add_argument('-s', '--subject',
                   help='Specific subject ID (e.g. subj001).')
    p.add_argument('-t', '--take',
                   help='Specific take ID (e.g. take002).')
    return p.parse_args()


def main():
    args = parse_args()
    sequences = list(iter_sequence_paths(args.data_dir, args.subject, args.take))
    if not sequences:
        filt = ' '.join(f'{k}={v}'
                        for k, v in [('subject', args.subject), ('take', args.take)] if v)
        raise SystemExit(
            f'No sequences found under {args.data_dir}'
            + (f' matching {filt}' if filt else '')
        )

    loop = args.subject is None and args.take is None
    if loop:
        random.shuffle(sequences)
        print(f'Playing {len(sequences)} sequences from {args.data_dir} '
              f'in random order (looping).')
    else:
        print(f'Playing {len(sequences)} sequence(s) from {args.data_dir}.')

    width = len(str(len(sequences)))
    while True:
        for i, (subj, take, path) in enumerate(sequences, 1):
            print(f'  [{i:>{width}}/{len(sequences)}] {subj}/{take}')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            visualize_sequence(data)
        if not loop:
            break
        random.shuffle(sequences)


if __name__ == '__main__':
    main()
