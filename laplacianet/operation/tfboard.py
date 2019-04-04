import threading
import os
def launchTensorBoard():
    os.system('tensorboard --logdir=' + './checkpoint/highlayer/lev_scale_4/' + ' --port=8888')
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()
