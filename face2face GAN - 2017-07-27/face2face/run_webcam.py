from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils.video import FPS
from threading import Thread


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('--show', dest='display_landmark', type=int, default=0, choices=[0, 1],
                    help='0 shows the normal input and 1 the facial landmark.')
parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, \
                    help='Face landmark model file.', default='shape_predictor_68_face_landmarks.dat')
parser.add_argument('--tf-model', dest='frozen_model_file', type=str, help='Frozen TensorFlow model file.')
parser.add_argument('-d', '--downsample', type=float, default=1, help='Downsample factor before face detection')
parser.add_argument('--video', type=str, help="Stream from input video file", default=None)
parser.add_argument('--video-out', type=str, help="Save to output video file", default=None)
parser.add_argument('--fps', type=int, help="Frames Per Second for output video file", default=10)
parser.add_argument('--skip', type=int, help="Speed up processing by skipping frames", default=0)
parser.add_argument('--no-gui', action='store_true', help="Don't render the gui", default=False)
parser.add_argument('--skip-fails', action='store_true', help="Don't render frames where no face is detected", default=False)
parser.add_argument('--scale', type=float, help="Scale the output image", default=1)
parser.add_argument('--points', action='store_true', help='Draw landmark points instead of lines', default=False)
parser.add_argument('--zoom', type=float, help="Zoom factor", default=1)
args = parser.parse_args()


CROP_SIZE = 256


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


def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))


def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main():
    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(graph=graph, config=config)

    if args.video is not None:
        cap = WebcamVideoStream(args.video).start()
    else:
        cap = WebcamVideoStream(args.video_source).start()

    if args.video_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.video_out, fourcc, args.fps, (int(512*args.scale),int(256*args.scale)))

    fps = FPS().start()

    landmark_toggle = False

    last_image = np.zeros((256,256), dtype=np.uint8)

    count = 0

    while(True):
        ret, frame = cap.read()

        if ret is True:       
            if args.skip and count % args.skip != 0:
                print("Skipping",count)
                continue
            else:
                if args.zoom > 1:
                    o_h, o_w, _ = frame.shape
                    frame = cv2.resize(frame, None, fx=args.zoom, fy=args.zoom)
                    h, w, _ = frame.shape
                    off_h, off_w = int((h - o_h) / 2), int((w - o_w) / 2)
                    frame = frame[off_h:h-off_h, off_w:w-off_w, :]


                if args.downsample > 1:
                    downsample_scale = args.downsample
                else:
                    # Auto-scale to CROP_SIZE
                    small_side = min(frame.shape[0], frame.shape[1])
                    downsample_scale = 1 / (CROP_SIZE / small_side)
                

                # resize image and detect face
                frame_resize = cv2.resize(frame, None, fx=1 / downsample_scale, fy=1 / downsample_scale)
                count += 1
                print("Frame:",count,"Shape:",frame_resize.shape)

                # print(frame_resize.shape)
                # print (frame_resize.shape)
                gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 1)
                black_image = np.zeros(frame_resize.shape, np.uint8)
                # black_image = np.zeros(frame.shape, np.uint8)

                for face in faces:
                    detected_landmarks = predictor(gray, face).parts()
                    landmarks = [[p.x, p.y] for p in detected_landmarks]
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

                # generate prediction
                if len(faces) > 0:
                    combined_image = np.concatenate([resize(black_image), resize(frame_resize)], axis=1)
                    image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
                    generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
                    image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
                    image_normal = np.concatenate([resize(frame_resize), image_bgr], axis=1)
                    image_landmark = np.concatenate([resize(black_image), image_bgr], axis=1)
                        
                    if landmark_toggle or args.display_landmark == 1:
                        img = image_landmark
                    else:
                        img = image_normal
                elif args.skip_fails:
                    # Don't show/write frames where no face is detected
                    continue
                else:
                    img = last_image

            last_image = img

            if args.scale != 1:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale)

            if args.video_out is not None:
                out.write(img)

            if args.no_gui is False:
                cv2.imshow('face2face', img)

            fps.update()

            key = cv2.waitKey(10) 
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('m'):
                landmark_toggle = not landmark_toggle
            
        else:
            # We're done here
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    sess.close()
    cap.stop()
    
    if args.video_out is not None:
        out.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
