# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    # ref_white = cv2.resize(cv2.imread("images_new/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    # ref_black = cv2.resize(cv2.imread("images_new/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    # ref_color = cv2.resize(cv2.imread("images_new/aligned001.jpg", cv2.IMREAD_COLOR), (0,0), fx=scale_factor,fy=scale_factor)

    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_color = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR), (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05 # a threshold for ON pixels
    ref_off = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2),cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
        # patt_gray = cv2.resize(cv2.imread("images_new/aligned%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        # populate scan_bits by putting the bit_code according to on_mask
        scan_bits[np.where(on_mask == True)] = scan_bits[np.where(on_mask == True)]+bit_code
        # print(np.count_nonzero(np.unique(scan_bits)))

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    camera_pointsC = []
    color_points=[]
    # print(ref_color.shape)

    # print(np.count_nonzero(np.unique(np.asarray(binary_codes_ids_codebook.values()))))
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            # use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # find for the camera (x,y) the projector (p_x, p_y).
            # store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            # print(binary_codes_ids_codebook.__len__())
            # print(scan_bits.max())

            x_p, y_p = binary_codes_ids_codebook[scan_bits[y, x]]
            if x_p >= 1279 or y_p >= 799:  # filter
                continue

            projector_points.append([x_p, y_p])
            camera_points.append([x/2, y/2])
            camera_pointsC.append([x, y])
            color_points.append(ref_color[y,x])


    # print(color_points[0])
    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    cimage= np.zeros((h,w,3), dtype=np.uint16)
    ppointsArr=np.asarray(projector_points,dtype=np.float32).reshape((projector_points.__len__(),1,2))
    cpointsArr=np.asarray(camera_points,dtype=np.float32).reshape((projector_points.__len__(),1,2))
    color_pointsArr=np.asarray(color_points,dtype=np.float32).reshape((projector_points.__len__(),3))

    # for i,elm in enumerate(camera_pointsC):
    #     x, y = elm
    #     valx, valy = projector_points[i]
    #     cimage[y,x,:]= valy*255/(valx+valy),valx*255/(valx+valy),0
    # plt.imshow(cimage)
    # plt.show()

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

    camera_normalized = cv2.undistortPoints(cpointsArr, camera_K, camera_d,P=camera_K)
    projector_normalized = cv2.undistortPoints(ppointsArr, projector_K, projector_d,P=projector_K)

    # print(projector_R)
    # print(projector_t)
    # print(projector_K)
    # print(camera_K)
    # print(projector_d)
    # print(camera_normalized[:10])
    # pProjMat = np.dot(projector_K,np.hstack((projector_R,projector_t)))
    pProjMat = np.dot(np.concatenate((projector_K.T,[[0,0,0]]),axis=0).T,
                         np.concatenate((np.concatenate((projector_R,projector_t),axis=1),[[0,0,0,1]]),axis=0))

    # print(pProjMat)
    # cProjMat = np.dot(camera_K, [[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    cProjMat = np.dot(np.concatenate((camera_K.T,[[0,0,0]]),axis=0).T,
                        np.eye(4,dtype=np.float32))
    # print(cProjMat)

    # use cv2.triangulatePoints to triangulate the normalized points
    triangulatedPoints=cv2.triangulatePoints(pProjMat,cProjMat,projector_normalized,camera_normalized)
    # use cv2.convertPointsFromHomogeneous to get real 3D points
    # print(triangulatedPoints.shape)
    # print(camera_normalized.shape)
    points_3d_u = cv2.convertPointsFromHomogeneous(triangulatedPoints.T)
    # name the resulted 3D points as "points_3d"
    mask = (points_3d_u[:, :, 2] > 200) & (points_3d_u[:, :, 2] < 1400)
    points_3d_wc = points_3d_u[np.where(mask[:, 0])]
    color_points=color_pointsArr[np.where(mask[:, 0])]
    # print(points_3d_wc.shape,color_points.shape)

    shape=points_3d_wc.shape
    points_3d = np.hstack((points_3d_wc.reshape(shape[0],-1),color_points[:,::-1])).reshape(shape[0],1,6)
    return points_3d

def write_3d_points(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
    # return points_3d, camera_points, projector_points
    return points_3d

def write_3d_points_withColor(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2],p[0, 3], p[0, 4], p[0, 5]))
    # return points_3d, camera_points, projector_points
    return points_3d

if __name__ == '__main__':


    # ===== DO NOT CHANGE THIS FUNCTION =====
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_3d_points_withColor(points_3d)
