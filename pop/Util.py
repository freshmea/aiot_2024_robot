from .__init__ import __main__
import cv2, ipywidgets as widgets
from IPython.display import display, clear_output
import librosa, math, os
import numpy as np
from tensorflow import one_hot as _one_hot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
       dim = None
       (h, w) = image.shape[:2]

       if width is None and height is None:
           return image

       if width is None:
           r = height / float(h)
           dim = (int(w * r), height)

       else:
           r = width / float(w)
           dim = (width, int(h * r))

       resized = cv2.resize(image, dim, interpolation = inter)

       return resized

windows={}

def imshow(title, image, width=300, height=300, mode='BGR'):
    import __main__
    global windows
    if not title in windows:
        windows[title] = {"widget":widgets.Image(width=width, height=height), "kernel":[]}
    elif windows[title]["widget"].width != width or windows[title]["widget"].height != height:
        if windows[title]["widget"].width != width:
            windows[title]["widget"].width = width

        if windows[title]["widget"].height != height:
            windows[title]["widget"].height = height

    _image=windows[title]["widget"]

    h, w = image.shape[:2]
    ih = int(_image.height)
    iw = int(_image.width)

    w_ratio = abs(h/w - ih/iw)
    h_ratio = abs(w/h - ih/iw)

    if w_ratio <= h_ratio:
        img = _image_resize(image, width=iw)
    else:
        img = _image_resize(image, height=ih)

    if mode.lower == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = bytes(cv2.imencode('.jpg', img)[1])
    _image.value=img

    kernel_num = len(__main__.In)-1

    if not kernel_num in windows[title]["kernel"]:
        windows[title]["kernel"].append(kernel_num)
        display(_image)

def enable_imshow():
    import __main__

    if "get_ipython" in dir(__main__):
        if not ("cv2" in dir(__main__)):
            import cv2
            __main__.cv2=cv2 

        __main__.cv2.imshow = imshow

def toMFCC(file_path, duration=1., rate=8000):
    wave, sr = librosa.load(file_path, mono=True, sr=rate)

    padlen=math.ceil(duration*(sr/int(0.048*sr)))
    
    wave = librosa.util.normalize(wave)
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=40, hop_length=int(0.048*sr), n_fft=int(0.096*sr))
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    pad_width = padlen - mfcc.shape[1]
    if pad_width>0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif pad_width<0:
        mfcc=mfcc[:,:padlen]
    
    return np.expand_dims(mfcc,-1)

def one_hot(num, maxlen):
    return _one_hot(num, maxlen)

def gstrmer(width=640, height=480, fps=30, flip=0):
    capture_width=width
    capture_height=height
    display_width=width
    display_height=height
    framerate=fps
    flip_method=flip
    
    return ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%s ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            __main__._camera_flip_method,
            display_width,
            display_height,
        )
    )

def createIMG(filename="img"):
    imgsize= (480,640,3)
    cv2.imwrite(filename+".jpg", np.reshape(np.frombuffer(imgcode,np.uint8), imgsize))