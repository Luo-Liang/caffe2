from caffe2.python import workspace, brew, model_helper

# This is probably the most reasonable single-GPU implementation of
# Alexnet. It's described by Krizhevsky here:
# https://arxiv.org/pdf/1404.5997v2.pdf and implemented here:
# https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-1gpu.cfg
# and is implemented in TensorFlow here: 
# https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/tutorials/image/alexnet/alexnet_benchmark.py
# and here:
# https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/tutorials/image/alexnet/alexnet_benchmark.py
# It expects images of size 224
def create_alexnet(
        model, data, num_input_channels, num_labels, is_test=False,
):
    conv1 = brew.conv(
        model,
        data,
        "conv1",
        num_input_channels, # dim_in
        64,                 # dim_out
        11,                 # kernel
        ('XavierFill', {}),
        ('ConstantFill', {}),
        stride=4,
        pad=2
    )
    relu1 = brew.relu(model, conv1, "conv1")
    norm1 = brew.lrn(model, relu1, "norm1", size=5, alpha=0.0001, beta=0.75)
    pool1 = brew.max_pool(model, norm1, "pool1", kernel=3, stride=2)
    conv2 = brew.conv(
        model,
        pool1,
        "conv2",
        64,
        192,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=2
    )
    relu2 = brew.relu(model, conv2, "conv2")
    norm2 = brew.lrn(model, relu2, "norm2", size=5, alpha=0.0001, beta=0.75)
    pool2 = brew.max_pool(model, norm2, "pool2", kernel=3, stride=2)
    conv3 = brew.conv(
        model,
        pool2,
        "conv3",
        192,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = brew.relu(model, conv3, "conv3")
    conv4 = brew.conv(
        model,
        relu3,
        "conv4",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu4 = brew.relu(model, conv4, "conv4")
    conv5 = brew.conv(
        model,
        relu4,
        "conv5",
        256,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu5 = brew.relu(model, conv5, "conv5")
    pool5 = brew.max_pool(model, relu5, "pool5", kernel=3, stride=2)
    fc6 = brew.fc(
        model,
        pool5, "fc6", 256 * 6 * 6, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu6 = brew.relu(model, fc6, "fc6")
    dropout1 = brew.dropout(model, relu6, 'dropout1', ratio=0.5, is_test=is_test)

    fc7 = brew.fc(
        model, dropout1, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu7 = brew.relu(model, fc7, "fc7")
    dropout2 = brew.dropout(model, relu7, 'dropout2', ratio=0.5, is_test=is_test)

    fc8 = brew.fc(
        model, dropout2, "fc8", 4096, num_labels, ('XavierFill', {}), ('ConstantFill', {})
    )
    # pred = brew.softmax(model, fc8, "pred")
    # xent = model.net.LabelCrossEntropy([pred, "label"], "xent")
    # model.net.AveragedLoss(xent, "loss")
    return fc8

# This one is closer to the MxNet implementation found here:
# https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/alexnet.py
def create_alexnetv0(
        model, data, num_input_channels, num_labels, is_test=False,
):
    conv1 = brew.conv(
        model,
        data,
        "conv1",
        num_input_channels, # dim_in
        96,                 # dim_out
        11,                 # kernel
        ('XavierFill', {}),
        ('ConstantFill', {}),
        stride=4,
        pad=2
    )
    relu1 = brew.relu(model, conv1, "conv1")
    norm1 = brew.lrn(model, relu1, "norm1", size=5, alpha=0.0001, beta=0.75)
    pool1 = brew.max_pool(model, norm1, "pool1", kernel=3, stride=2)
    conv2 = brew.conv(
        model,
        pool1,
        "conv2",
        96,
        256,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=2
    )
    relu2 = brew.relu(model, conv2, "conv2")
    norm2 = brew.lrn(model, relu2, "norm2", size=5, alpha=0.0001, beta=0.75)
    pool2 = brew.max_pool(model, norm2, "pool2", kernel=3, stride=2)
    conv3 = brew.conv(
        model,
        pool2,
        "conv3",
        256,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = brew.relu(model, conv3, "conv3")
    conv4 = brew.conv(
        model,
        relu3,
        "conv4",
        384,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu4 = brew.relu(model, conv4, "conv4")
    conv5 = brew.conv(
        model,
        relu4,
        "conv5",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu5 = brew.relu(model, conv5, "conv5")
    pool5 = brew.max_pool(model, relu5, "pool5", kernel=3, stride=2)
    fc6 = brew.fc(
        model,
        pool5, "fc6", 256 * 6 * 6, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu6 = brew.relu(model, fc6, "fc6")
    dropout1 = brew.dropout(model, relu6, 'dropout1', ratio=0.5, is_test=is_test)

    fc7 = brew.fc(
        model, dropout1, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu7 = brew.relu(model, fc7, "fc7")
    dropout2 = brew.dropout(model, relu7, 'dropout2', ratio=0.5, is_test=is_test)

    fc8 = brew.fc(
        model, dropout2, "fc8", 4096, num_labels, ('XavierFill', {}), ('ConstantFill', {})
    )
    return fc8
