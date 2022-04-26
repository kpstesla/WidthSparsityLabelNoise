import timm


def inception_resnet_v2(width, num_classes):
    return timm.models.InceptionResnetV2(num_classes=num_classes)
