import cv2
vidcap = cv2.VideoCapture('16-01-09.svo.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1