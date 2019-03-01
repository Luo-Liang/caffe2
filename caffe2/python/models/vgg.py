from caffe2.python import workspace, brew, model_helper

# This combines the Caffe2 implementation here:
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/convnet_benchmarks.py
# with the MxNet implementation here
# https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/vgg.py
# to better match the paper here: https://arxiv.org/pdf/1409.1556.pdf
def create_vgg(
        model, data, num_input_channels, num_labels, num_layers=11, is_test=False,
):

    if num_layers == 11: # VGG configuration A
        first_layers_count = 1
        last_layers_count = 2
    elif num_layers == 13: # VGG configuration D
        first_layers_count = 2
        last_layers_count = 2
    elif num_layers == 16: # VGG configuration D
        first_layers_count = 2
        last_layers_count = 3
    elif num_layers == 19: # VGG configuration E
        first_layers_count = 2
        last_layers_count = 4
    else:
        raise NotImplementedError("not currently supported: try one of {11, 13, 16, 19}, corresponding to VGG A, B, D, and E.")

    conv1 = brew.conv(
        model,
        data,
        "conv1",
        num_input_channels,
        64,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1,
    )
    relu1 = brew.relu(model, conv1, "conv1")
    for i in range(0, first_layers_count-1):
        conv1 = brew.conv(
            model,
            relu1,
            "conv1{}".format(i),
            64,
            64,
            3,
            ('XavierFill', {}),
            ('ConstantFill', {}),
            pad=1,
        )
        relu1 = brew.relu(model, conv1, "conv1{}".format(i))
        
    pool1 = brew.max_pool(model, relu1, "pool1", kernel=2, stride=2)
    conv2 = brew.conv(
        model,
        pool1,
        "conv2",
        64,
        128,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1,
    )
    relu2 = brew.relu(model, conv2, "conv2")
    for i in range(0, first_layers_count-1):
        conv2 = brew.conv(
            model,
            relu2,
            "conv2{}".format(i),
            128,
            128,
            3,
            ('XavierFill', {}),
            ('ConstantFill', {}),
            pad=1,
        )
        relu2 = brew.relu(model, conv2, "conv2{}".format(i))

    pool2 = brew.max_pool(model, relu2, "pool2", kernel=2, stride=2)
    conv3 = brew.conv(
        model,
        pool2,
        "conv3",
        128,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1,
    )
    relu3 = brew.relu(model, conv3, "conv3")
    for i in range(0, last_layers_count-1):
        conv4 = brew.conv(
            model,
            relu3,
            "conv4{}".format(i),
            256,
            256,
            3,
            ('XavierFill', {}),
            ('ConstantFill', {}),
            pad=1,
        )
        relu4 = brew.relu(model, conv4, "conv4{}".format(i))
    pool4 = brew.max_pool(model, relu4, "pool4", kernel=2, stride=2)
    conv5 = brew.conv(
        model,
        pool4,
        "conv5",
        256,
        512,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1,
    )
    relu5 = brew.relu(model, conv5, "conv5")
    for i in range(0, last_layers_count-1):
        conv6 = brew.conv(
            model,
            relu5,
            "conv6{}".format(i),
            512,
            512,
            3,
            ('XavierFill', {}),
            ('ConstantFill', {}),
            pad=1,
        )
        relu6 = brew.relu(model, conv6, "conv6{}".format(i))
    pool6 = brew.max_pool(model, relu6, "pool6", kernel=2, stride=2)
    conv7 = brew.conv(
        model,
        pool6,
        "conv7",
        512,
        512,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1,
    )
    relu7 = brew.relu(model, conv7, "conv7")
    for i in range(0, last_layers_count-1):
        conv8 = brew.conv(
            model,
            relu7,
            "conv8{}".format(i),
            512,
            512,
            3,
            ('XavierFill', {}),
            ('ConstantFill', {}),
            pad=1,
        )
        relu8 = brew.relu(model, conv8, "conv8{}".format(i))
    pool8 = brew.max_pool(model, relu8, "pool8", kernel=2, stride=2)

    fcix = brew.fc(
        model, pool8, "fcix", 512 * 7 * 7, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    reluix = brew.relu(model, fcix, "fcix")
    fcx = brew.fc(
        model, reluix, "fcx", 4096, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relux = brew.relu(model, fcx, "fcx")
    fcxi = brew.fc(
        model, relux, "fcxi", 4096, num_labels, ('XavierFill', {}),
        ('ConstantFill', {})
    )

    return fcxi
