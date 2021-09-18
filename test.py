import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

number_of_fingers_up = 0

# Currently only works when left hand is used and palm is facing the camera
# To improve on this, detection of which hand and also wether the palm is facing the camera or not it required
# For which hand is being used, it can be taken from results.multi_handedness and
# to detect wether palm is facing or not, the position of pinky finger and thumb can be used, which is on left and right

def which_hand(multihand, ix):
  if multihand != None:
    return multihand[ix].classification[0].label
  else:
    return None

def is_palm(thumb, pinky, which_hand):
  if which_hand == "Left":
    if thumb.x > pinky.x:
      return True
    else:
      return False
  elif which_hand == "Right":
    if thumb.x < pinky.x:
      return True
    else:
      return False

def is_finger_up( nfing , result , has_hand, whand, ipalm):
    if nfing != 0: # Finger detection
        fstartn = 5 + ( 4*(nfing-1) )
        npoints = [ result[0], result[fstartn], result[fstartn+1], result[fstartn+2], result[fstartn+3] ]
        if npoints[1].y < npoints[0].y and npoints[2].y < npoints[1].y and npoints[3].y < npoints[2].y:
            return True
    else: # Thumb detection
        fstartn = 2
        npoints = [ result[0], result[fstartn], result[fstartn+1], result[fstartn+2] ]
        if whand == "Left" and ipalm == True:
            if npoints[1].y < npoints[0].y and npoints[2].y < npoints[1].y and npoints[3].y < npoints[2].y and npoints[3].x > npoints[1].x and npoints[3].x > npoints[2].x and npoints[2].x > npoints[1].x:
                return True
        elif whand == "Left" and ipalm == False:
            if npoints[1].y < npoints[0].y and npoints[2].y < npoints[1].y and npoints[3].y < npoints[2].y and npoints[3].x < npoints[1].x and npoints[3].x < npoints[2].x and npoints[2].x < npoints[1].x:
                return True
        elif whand == "Right" and ipalm == True:
            if npoints[1].y < npoints[0].y and npoints[2].y < npoints[1].y and npoints[3].y < npoints[2].y and npoints[3].x < npoints[1].x and npoints[3].x < npoints[2].x and npoints[2].x < npoints[1].x:
                return True
        elif whand == "Right" and ipalm == False:
            if npoints[1].y < npoints[0].y and npoints[2].y < npoints[1].y and npoints[3].y < npoints[2].y and npoints[3].x > npoints[1].x and npoints[3].x > npoints[2].x and npoints[2].x > npoints[1].x:
                return True
    return False

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    
    number_of_fingers_up = 0
    u = -1
    if results.multi_handedness != None and results.multi_hand_landmarks != None:
      for x in results.multi_handedness:
          u += 1
          i = 0
          whand = which_hand(results.multi_handedness, u)
          ipalm = is_palm(results.multi_hand_landmarks[u].landmark[2], results.multi_hand_landmarks[u].landmark[17], whand)
          while i < 5:
              isit = is_finger_up( i, results.multi_hand_landmarks[u].landmark , results.multi_handedness, whand, ipalm)
              if isit:
                  number_of_fingers_up += 1
              i = i + 1
    print("Number of fingers+thumbs up: " + str(number_of_fingers_up))

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
