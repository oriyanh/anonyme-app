from keras_vggface.vggface import VGGFace

def get_vggface_model():
    return VGGFace(model='resnet50')