from utils import read_video, save_video
import cv2
import numpy as np

video_frames = read_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/input_videos/fouda.mp4')
# np.save('fouda.npy', video_frames)
# video_frames = np.load('fouda.npy', allow_pickle=True)

out_frames = []
# for frame in out_frames:
gray_frame = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# output_frame = video_frames[0].copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(x2 - x1) < 10: 
            cv2.line(video_frames[0], (x1, y1), (x2, y2), (120, 120, 0), 2)

# save_video('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/output_videos/fouda.mp4', video_frames)
cv2.imwrite('C:/Users/moham/Desktop/GP/Computer_Vision_Fifa/output_videos/line_test.png', video_frames[0])
















# [{}, {}, 
{1: {'bbox': [1040.106201171875, 331.9769287109375, 1061.189697265625, 352.4671325683594], 'position': (1050, 342), 'position_adjusted': (np.float32(1040.9014), np.float32(342.63138)), 'position_transformed': [54.2136116027832, 10.469291687011719]}}, 
# {1: {'bbox': [1040.2127685546875, 331.8479919433594, 1061.18017578125, 352.3900451660156], 'position': (1050, 342), 'position_adjusted': (1050, 342), 'position_transformed': [54.6875, 10.449999809265137]}}, 
# {1: {'bbox': [1038.4998779296875, 332.9436950683594, 1061.323486328125, 354.0328674316406], 'position': (1049, 343), 'position_adjusted': (1049, 343), 'position_transformed': [54.63541793823242, 10.480555534362793]}}, 
# {1: {'bbox': [1035.0234375, 333.6800537109375, 1058.9998779296875, 357.0700378417969], 'position': (1047, 345), 'position_adjusted': (np.float32(1035.3213), np.float32(342.8008)), 'position_transformed': [53.9229850769043, 10.474469184875488]}}, 
# {1: {'bbox': [1031.1622314453125, 339.8634948730469, 1055.2222900390625, 361.77996826171875], 'position': (1043, 350), 'position_adjusted': (np.float32(1048.491), np.float32(347.79593)), 'position_transformed': [54.60890579223633, 10.627098083496094]}}, 
# {1: {'bbox': [1028.405029296875, 343.8585205078125, 1047.7486572265625, 362.26666259765625], 'position': (1038, 353), 'position_adjusted': (np.float32(1046.8154), np.float32(353.0113)), 'position_transformed': [54.521636962890625, 10.786456108093262]}}, 
# {1: {'bbox': [1028.4434814453125, 343.8620910644531, 1047.7083740234375, 362.2810363769531], 'position': (1038, 353), 'position_adjusted': (1038, 353), 'position_transformed': [54.0625, 10.786110877990723]}}, 
# {1: {'bbox': [1028.7828369140625, 344.7818603515625, 1048.3809814453125, 362.7991638183594], 'position': (1038, 353), 'position_adjusted': (np.float32(1041.2256), np.float32(358.64996)), 'position_transformed': [54.230499267578125, 10.958748817443848]}}, 
# # {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
# {1: {'bbox': [1057.5692138671875, 362.4621276855469, 1081.5380859375, 387.2341003417969], 'position': (1069, 374), 'position_adjusted': (np.float32(1080.8596), np.float32(374.04227)), 'position_transformed': [56.29477310180664, 11.429069519042969]}}, 
# {1: {'bbox': [1058.19775390625, 363.7891845703125, 1082.8228759765625, 387.3988037109375], 'position': (1070, 375), 'position_adjusted': (np.float32(1079.065), np.float32(374.33292)), 'position_transformed': [56.201297760009766, 11.437950134277344]}}, 
# {1: {'bbox': [1068.158935546875, 362.4766845703125, 1088.56591796875, 380.63287353515625], 'position': (1078, 371), 'position_adjusted': (np.float32(1102.9487), np.float32(367.2277)), 'position_transformed': [57.445247650146484, 11.220846176147461]}}, 
# {1: {'bbox': [1068.6453857421875, 362.57391357421875, 1088.82421875, 380.5770263671875], 'position': (1078, 371), 'position_adjusted': (1078, 371), 'position_transformed': [56.14583206176758, 11.336111068725586]}},
# {1: {'bbox': [1075.58203125, 364.6285705566406, 1099.2476806640625, 385.5549621582031], 'position': (1087, 375), 'position_adjusted': (np.float32(1095.1041), np.float32(372.57764)), 'position_transformed': [57.03667449951172, 11.384316444396973]}},
# {1: {'bbox': [1080.5850830078125, 364.541748046875, 1103.7755126953125, 385.30865478515625], 'position': (1092, 374), 'position_adjusted': (np.float32(1100.2083), np.float32(377.05746)), 'position_transformed': [57.302513122558594, 11.521200180053711]}}, 
# {1: {'bbox': [1080.6871337890625, 364.4759216308594, 1103.7489013671875, 385.30535888671875], 'position': (1092, 374), 'position_adjusted': (1092, 374), 'position_transformed': [56.875, 11.427778244018555]}}, {}, {}, {}]