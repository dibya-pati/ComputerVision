import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

#changed
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.002)


def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # print(gray)
    # cv2.imshow('grs',gray)
    # cv2.waitKey(0)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def cmShift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()

    #debug
    # cv2.imshow('frame',frame)
    # cv2.waitKey(0)
    # print(type(frame),frame)

    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame

    pt_x, pt_y=c + w/2.0,r + h/2.0
    #channged
    output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        retval,track_window=cv2.CamShift(prob,track_window,term_crit)

        # Draw it on image
        # print(retval)
        pts = cv2.boxPoints(retval)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        c,r,w,h=track_window

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,c + w/2.0,r + h/2.0)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]


def particle_tracker(v, file_name):
    # Open output file

    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    stepsize= 1
    # read first frame
    ret, frame = v.read()

    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame

    pt_x, pt_y=c + w/2.0,r + h/2.0
    #channged
    output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    n_particles = 450

    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    prob = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # print(init_pos.shape)
    # print(particles.shape)
    f0 = particleevaluator(prob, particles.T) * np.ones(n_particles)  # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles  # weights are uniform (at first)
    # print(np.average(particles,axis=0))
    pos=init_pos

    while(1):
        stepsize=30
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
        # c,r,w,h: obtain using detect_one_face()
        # Particle motion model: uniform step (TODO: find a better motion model)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #moving in the direction of particle with maximum weight
        # x=pos[0]+np.random.uniform(-5, 5)
        # y=pos[1]+np.random.uniform(-5, 5)
        # init_pos=pos
        np.add(particles, np.random.uniform(-stepsize,stepsize, particles.shape), out=particles, casting="unsafe")

        # print(particles.shape)
        # print(np.argmax())
        # Clip out-of-bounds particles,determine the width and height and clip to this rnge
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0])) - 1).astype(int)

        f = particleevaluator(prob, particles.T)  # Evaluate particles
        weights = np.float32(f.clip(1))  # Weight ~ histogram response
        weights /= np.sum(weights)  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average

        # print(f-f.clip(1))

        if 1. / np.sum(weights ** 2) < n_particles / 2:  # If particle cloud degenerate:
            # print('resampled')
            particles = particles[resample(weights), :]  # Resample particles according to weights
        # resample() function is provided for you

        img2 = cv2.drawMarker(frame,(pos[0],pos[1]),(0,255,0),markerType=1,markerSize=10)
        cv2.imshow('img',img2)
        cv2.waitKey(60)

        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # # plt.scatter(particles[1],particles[0], c='b', s=5)
        # plt.pause(0.05)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def kf_tracker(v, file_name):
    # Open output file

    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    stepsize= 1
    # read first frame
    ret, frame = v.read()

    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame

    pt_x, pt_y=c + w/2.0,r + h/2.0
    output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype=np.float32)  # initial position
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
    kalman.errorCovPost = 1e-1 * np.eye(4, 4,dtype=np.float32)
    kalman.statePost = state
    # print(kalman.measurementMatrix)

    while(1):
        ret, frame = v.read() # read another frame
        if ret == False:
            break
        c, r, w, h = detect_one_face(frame)

        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
        # c,r,w,h: obtain using detect_one_face()
        # Particle motion model: uniform step (TODO: find a better motion model)

        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # prob = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # retval,track_window=cv2.CamShift(prob,track_window,term_crit)
        # c, r, w, h = track_window

        prediction = kalman.predict()
        pos=prediction[:2]

        # if not((c==0)and(r==0)and(w==0)and(h==0)):  # e.g. face found
        if (any([c,r,w,h])>0):  # e.g. face found
            posterior = kalman.correct(np.array([[np.float32(c+w/2)],[np.float32(r+h/2)]]))
            # print(posterior[:2])
            pos = posterior[:2]
        else:
            pass
            # print('entered kalman',frameCounter)

        # use prediction or posterior as your tracking result

        img2 = cv2.drawMarker(frame,(pos[0],pos[1]),(0,255,0),markerType=1,markerSize=10)
        cv2.imshow('img',img2)
        cv2.waitKey(60)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

def of_tracker(v, file_name):
    # Open output file

    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()

    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # print(frame.shape)
    mask = np.zeros_like(frame)

    facemask = np.zeros(shape=(frame.shape[0],frame.shape[1]),dtype=np.uint8)
    facemask[r+5:r + h-5, c+8:c + w-8] = 255

    # cv2.imshow('mask',facemask)
    # cv2.waitKey(0)
    lk_params = dict(winSize=(10, 10),
                     maxLevel=6,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0003))
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.2,
                          minDistance=9,
                          blockSize=15)

    # cv2.imshow('face',frame[r:r+h,c:c+w])
    # cv2.waitKey(0)

    # Write track point for first frame
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray,mask=facemask , **feature_params)

    pt_x, pt_y=c + w/2.0,r + h/2.0
    output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    color = np.random.randint(0, 255, (100, 3))

    while(1):
        ret, frame = v.read() # read another frame
        if ret == False:
            break
        c, r, w, h = detect_one_face(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # use prediction or posterior as your tracking result
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # print(good_new.shape)
        pos =(c+w/2,r+h/2)
        if (all([c,r,w,h])==0):  # e.g. face not found
            pos = np.average(good_new,axis=0)
            # print(frameCounter,'using optical flow')
            # print(pos)
        else:
            pass
            # print('entered kalman',frameCounter)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    output.close()


if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        cmShift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kf_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
