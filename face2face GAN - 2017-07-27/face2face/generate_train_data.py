from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video
import random
from threading import Thread


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--downsample', type=float, default=3, help='Downsample factor')
parser.add_argument('-dmax', type=float, default=0, help='Max randomized downsample factor')
parser.add_argument('-dmin', type=float, default=0, help='Min randomized downsample factor')
parser.add_argument('-n', type=int, default=1, help='Number of samples to generate per frame')
parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
parser.add_argument('--num', type=int, help='Number of train data to be created.')
parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
parser.add_argument('--points', action='store_true', help='Draw landmarks as circles instead of polylines', default=False)
parser.add_argument('--zoom', type=float, help="Zoom factor", default=1)
parser.add_argument('--webcam', action='store_true', help="Take input from webcam instead of video file", default=False)
args = parser.parse_args()


class WebcamVideoStream:
    # From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.ret, self.frame) = self.stream.read()
 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.ret, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return (self.ret, self.frame)
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        self.stop()


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def main():
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)

    if args.webcam:
        cap = WebcamVideoStream(0).start()
    else:
        cap = cv2.VideoCapture(args.filename)

    fps = video.FPS().start()

    count = 0
    frame_count = 0

    ret = True

    while ret is True:
        frame_count += 1
        print("Frame:",frame_count)

        ret, frame = cap.read()

        if args.zoom > 1:
            o_h, o_w, _ = frame.shape
            frame = cv2.resize(frame, None, fx=args.zoom, fy=args.zoom)
            h, w, _ = frame.shape
            off_h, off_w = int((h - o_h) / 2), int((w - o_w) / 2)
            frame = frame[off_h:h-off_h, off_w:w-off_w, :]

        for _ in range(args.n):
            if args.dmax > 0 and args.dmin > 0:
                down_scale = np.random.uniform(args.dmin, args.dmax)
            else:
                down_scale = args.downsample
            
            down = 1 / down_scale

            frame_resize = cv2.resize(frame, None, fx=down, fy=down)
            gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            black_image = np.zeros(frame_resize.shape, np.uint8)

            t = time.time()

            # Perform if there is a face detected
            if len(faces) == 1:
                # Display the resulting frame
                count += 1
                print(count)
                    
                for face in faces:
                        detected_landmarks = predictor(gray, face).parts()
                        landmarks = [[int(p.x), int(p.y)] for p in detected_landmarks]

                        color = (255, 255, 255)
                        thickness = 3

                        if args.points:
                            jaw = landmarks[0:17]
                            left_eyebrow = landmarks[22:27]
                            right_eyebrow = landmarks[17:22]
                            nose_bridge = landmarks[27:31]
                            lower_nose = landmarks[30:35]
                            left_eye = landmarks[42:48]
                            right_eye = landmarks[36:42]
                            outer_lip = landmarks[48:60]
                            inner_lip = landmarks[60:68]
                            for part in [jaw, left_eyebrow, right_eyebrow, nose_bridge, lower_nose, left_eye, right_eye, outer_lip, inner_lip]:
                                for x,y in part:
                                    cv2.circle(black_image, (x, y), 1, (255, 255, 255), -1)
                        else:
                            jaw = reshape_for_polyline(landmarks[0:17])
                            left_eyebrow = reshape_for_polyline(landmarks[22:27])
                            right_eyebrow = reshape_for_polyline(landmarks[17:22])
                            nose_bridge = reshape_for_polyline(landmarks[27:31])
                            lower_nose = reshape_for_polyline(landmarks[30:35])
                            left_eye = reshape_for_polyline(landmarks[42:48])
                            right_eye = reshape_for_polyline(landmarks[36:42])
                            outer_lip = reshape_for_polyline(landmarks[48:60])
                            inner_lip = reshape_for_polyline(landmarks[60:68])
                            cv2.polylines(black_image, [jaw], False, color, thickness)
                            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
                            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
                            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
                            cv2.polylines(black_image, [lower_nose], True, color, thickness)
                            cv2.polylines(black_image, [left_eye], True, color, thickness)
                            cv2.polylines(black_image, [right_eye], True, color, thickness)
                            cv2.polylines(black_image, [outer_lip], True, color, thickness)
                            cv2.polylines(black_image, [inner_lip], True, color, thickness)

                cv2.imwrite("original/{}_{}.png".format(count, round(down, 3)), frame)
                cv2.imwrite("landmarks/{}_{}.png".format(count, round(down, 3)), black_image)
                fps.update()

                print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
            else:
                print("No face detected")

        if count == args.num:
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()

