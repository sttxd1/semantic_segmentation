from deeplab_v3_plus.models.backbone import resnet

from deeplab_v3_plus.models.backbone import mobilenetv2


def build_backbone(backbone, output_stride):
    """

    Args:
        backbone (str): name of the backbone
        output_stride (int): The ratio of input image spatial resolution to the final output resolution
    """
    if backbone in resnet.__all__:
        if output_stride == 16:
            replace_stride_with_dilation = (False, False, True)
        elif output_stride == 8:
            replace_stride_with_dilation = (False, True, True)
        else:
            raise NotImplementedError

        model = getattr(resnet, backbone)
        return model(pretrained=True, progress=True, replace_stride_with_dilation=replace_stride_with_dilation)
    

    elif backbone in mobilenetv2.__all__:
        if output_stride == 16:
            output_stride = 16  # MobileNetV2 handles its own stride internally
        elif output_stride == 8:
            output_stride = 8  # MobileNetV2 handles its own stride internally
        else:
            raise NotImplementedError

        model = getattr(mobilenetv2, backbone)
        return model(pretrained=True, progress=True, output_stride=output_stride)

    else:
        raise NotImplementedError
