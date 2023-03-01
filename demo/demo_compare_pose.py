# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import torch
import numpy as np
import cv2
import argparse
import json
import pickle
from bodymocap.models import SMPL

############# input parameters  #############
from bodymocap.core import config 
from renderer import viewer2D#, glViewer
from renderer.visualizer import Visualizer

from bodymocap.utils.timer import Timer
g_timer = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('--mocapdir', type=str, default=None, help='Folder of output images.')

import numpy as np

def orientation_diff(R1, R2):
    """
    Computes the difference between two 3D orientation matrices.

    Parameters:
    R1 (ndarray): A 3x3 rotation matrix representing the first orientation.
    R2 (ndarray): A 3x3 rotation matrix representing the second orientation.

    Returns:
    float: The angle between the two orientations in radians.
    """

    # Compute the rotation matrix that takes R1 to R2
    R12 = np.dot(R2, R1.T)

    # Compute the axis and angle of the rotation that takes R1 to R2
    temp_val = (np.trace(R12) - 1) / 2
    if temp_val > 1:
        temp_val = 1
    angle = np.arccos(temp_val)
    axis = np.array([R12[2, 1] - R12[1, 2], R12[0, 2] - R12[2, 0], R12[1, 0] - R12[0, 1]])
    

    # Normalize the axis vector
    if (np.linalg.norm(axis) != 0):
        axis /= np.linalg.norm(axis)

    # Compute the angle difference
    angle_diff = angle * np.sign(np.dot(axis, np.array([0, 0, 1])))

    return angle_diff

def Joints_rot_match(joints_rotmat1, joints_rotmat2):
    thresh = 0.1
    i = 0
    for joint1, joint2 in zip(joints_rotmat1, joints_rotmat2):
        # joint1, joint2: 3x3 matrix to represent a joint's orientation
        if (orientation_diff(joint1, joint2) > thresh):
            return False
    
    return True

def get_video_path(args):
    if args.webcam:
        video_path = 0
    elif args.url:
        if args.download:
            os.makedirs("./webvideos",exist_ok=True)
            downloadPath ="./webvideos/{0}.mp4".format(os.path.basename(args.url))
            cmd_download = "youtube-dl -f best {0} -o {1}".format(args.url,downloadPath)
            print(">> Downloading: {}".format(args.url))
            print(">> {}".format(cmd_download))
            #download via youtube-dl
            os.system(cmd_download)
            video_path = downloadPath
        else:
            try:
                import pafy
                url = args.url #'https://www.youtube.com/watch?v=c5nhWy7Zoxg'
                vPafy = pafy.new(url)
                play = vPafy.getbest(preftype="webm")
                video_path = play.url
                video_path = url
            except:
                video_path = args.url
    elif args.vPath:
        video_path = args.vPath
    else:
        assert False
    return video_path

def RunMonomocap(args, smpl, mocapDir, visualizer):
    fileNames = sorted(os.listdir(mocapDir))

    print(f"Loading mocap data from: {mocapDir}")
    mocapDataAll = []
    for fname in fileNames:

        filePath = os.path.join(mocapDir, fname)

        with open(filePath,'rb') as f:
            mocapData_frame = pickle.load(f)

        mocapDataAll.append(mocapData_frame)

    frameNum = len(mocapDataAll)
    frameIter =0
    assert len(mocapDataAll)>0
    while(True):#for mocapData_frame in mocapDataAll:

        print(f"frameIdx:{frameIter} / {frameNum} ")
        mocapData_frame = mocapDataAll[frameIter]

        frameIter+=1
        if frameIter >=frameNum:
            frameIter = 0    #Looping

        # personNum = len(mocapData_frame)
        for mocapData in mocapData_frame:

            pred_betas = torch.from_numpy( mocapData['parm_shape'][np.newaxis,:])    #(10,)
            pred_rotmat = torch.from_numpy( mocapData['parm_pose'][np.newaxis,:])   #24x3x3
            pred_vertices_imgspace =mocapData['pred_vertices_imgspace']
            pred_joints_imgspace =mocapData['pred_joints_imgspace']


            joints_rotmat = pred_rotmat[0].numpy() # 3D rotation matrix for 24 joints
            if (Joints_rot_match(joints_rotmat, joints_rotmat)):
                print("Match!!!")
            else:
                print("Not Match!!!")

            if False:    #One way to visualize SMPL from saved vertices
                tempMesh = {'ver': pred_vertices_imgspace, 'f':  smpl.faces}
                meshList=[]
                skelList=[]
                meshList.append(tempMesh)
                skelList.append(pred_joints_imgspace.ravel()[:,np.newaxis])  #(49x3, 1)

                visualizer.visualize_gui_naive(meshList, skelList)

            elif False: #Alternative way from SMPL parameters
                pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0] ], pose2rot=False)
                pred_vertices = pred_output.vertices
                # pred_joints_3d = pred_output.joints
                pred_vertices = pred_vertices[0].cpu().numpy()
                
                tempMesh = {'ver': pred_vertices_imgspace, 'f':  smpl.faces}
                meshList=[]
                skelList=[]
                # bboxXYWH_list=[]
                meshList.append(tempMesh)
                skelList.append(pred_joints_imgspace.ravel()[:,np.newaxis])  #(49x3, 1)
                visualizer.visualize_gui_naive(meshList, skelList)

            else: #Another alternative way using a funtion
                smpl_pose_list =  [ pred_rotmat[0].numpy() ]        #build a numpy array
                visualizer.visualize_gui_smplpose_basic(smpl, smpl_pose_list ,isRotMat=True )       #Assuming zero beta

        # g_timer.toc(average =True, bPrint=True,title="Detect+Regress+Vis")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # checkpoint = args.checkpoint
    # video_path = get_video_path(args)
    mocapDir = args.mocapdir

    visualizer = Visualizer('gui')

    smplModelPath = './extradata/smpl//basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to('cpu')

    RunMonomocap(args, smpl, mocapDir, visualizer)