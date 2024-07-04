import cv2
import numpy as np
import Vehicle

# Counter variables
cnt_up = 0
cnt_down = 0
UpMTR = 0
DownMTR = 0
UpLV = 0
UpHV = 0
DownLV = 0
DownHV = 0

# Set input video
video_path = 'test1.MP4'
cap = cv2.VideoCapture(r"C:\Users\ASUS\Downloads\vehicle-counting-and-classification-opencv-master\vehicle-counting-and-classification-opencv-master\tes2.MP4")

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Capture the properties of VideoCapture to console
for i in range(19):
    print(i, cap.get(i))

# Get width and height of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("Error: Could not get video dimensions.")
    exit()

frameArea = height * width
areaTH = frameArea / 800

# Input/Output Lines
font = cv2.FONT_HERSHEY_SIMPLEX
line_up_color = (0, 0, 255)
line_down_color = (255, 0, 0)
line_up = int(2 * (height / 5))
line_down = int(3 * (height / 5))
up_limit = int(1 * (height / 5))
down_limit = int(4 * (height / 5))

# Coordinates for lines
pt1 = [0, line_down]
pt2 = [width, line_down]
pts_L1 = np.array([pt1, pt2], np.int32).reshape((-1, 1, 2))

pt3 = [0, line_up]
pt4 = [width, line_up]
pts_L2 = np.array([pt3, pt4], np.int32).reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [width, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32).reshape((-1, 1, 2))

pt7 = [0, down_limit]
pt8 = [width, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32).reshape((-1, 1, 2))

# Create the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
vehicles = []
max_p_age = 5
pid = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('EOF')
        break

    for vehicle in vehicles:
        vehicle.age_one()

    fgmask = fgbg.apply(frame)

    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
    except Exception as e:
        print('Error:', e)
        break

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new_vehicle = True
            for vehicle in vehicles:
                if abs(x - vehicle.getX()) <= w and abs(y - vehicle.getY()) <= h:
                    new_vehicle = False
                    vehicle.updateCoords(cx, cy)
                    if vehicle.going_UP(line_down, line_up):
                        vehicle.state = '1'
                        roi = frame[y:y + h, x:x + w]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        height = h
                        width = w
                        kll = 2 * (height + width)
                        if kll < 300:
                            UpMTR += 1
                        elif kll < 500:
                            UpLV += 1
                        elif kll > 500:
                            UpHV += 1
                        cnt_up += 1
                    elif vehicle.going_DOWN(line_down, line_up):
                        vehicle.state = '1'
                        roi = frame[y:y + h, x:x + w]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        height = y + h
                        width = x + w
                        luas = height * width
                        if luas < 600000:
                            DownLV += 1
                        elif luas > 600000:
                            DownHV += 1
                        # Ensure DownMTR is incremented
                        DownMTR += 1
                        cnt_down += 1
                    break
                if vehicle.getState() == '1':
                    if vehicle.getDir() == 'down' and vehicle.getY() > down_limit:
                        vehicle.setDone()
                    elif vehicle.getDir() == 'up' and vehicle.getY() < up_limit:
                        vehicle.setDone()
                if vehicle.timedOut():
                    vehicles.remove(vehicle)
            if new_vehicle:
                new_vehicle = Vehicle.MyVehicle(pid, cx, cy, max_p_age)
                vehicles.append(new_vehicle)
                pid += 1

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for vehicle in vehicles:
        cv2.putText(frame, str(vehicle.getId()), (vehicle.getX(), vehicle.getY()), font, 0.3, vehicle.getRGB(), 1, cv2.LINE_AA)

    frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

    MTR_up = f'Up Motor: {UpMTR}'
    MTR_down = f'Down Motor: {DownMTR}'
    LV_up = f'Up Car: {UpLV}'
    HV_up = f'Up Truck/Bus: {UpHV}'
    LV_down = f'Down Car: {DownLV}'
    HV_down = f'Down Truck/Bus: {DownHV}'

    cv2.putText(frame, MTR_up, (10, 40), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, MTR_up, (10, 40), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, LV_up, (10, 90), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, LV_up, (10, 90), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, HV_up, (10, 140), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, HV_up, (10, 140), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, MTR_down, (10, 190), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, MTR_down, (10, 190), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, LV_down, (10, 240), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, LV_down, (10, 240), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, HV_down, (10, 290), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, HV_down, (10, 290), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Frame', cv2.resize(frame, (800, 600)))

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
