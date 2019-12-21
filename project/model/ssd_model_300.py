from keras.applications import VGG16
from keras.models import Model
from keras.initializers import RandomUniform

from keras.regularizers import l2

from keras import layers

VGG = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
BASE_MODEL = Model(inputs=VGG.input, outputs=VGG.get_layer('block5_conv3').output)


initializer = RandomUniform(minval=-1e-2, maxval=1e-2)
# activation = layers.LeakyReLU(alpha=0.1)
activation = 'relu'


def ssd_model_300(reg=0.00003):
    conv4_3 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv4_3_padding')(BASE_MODEL.layers[-5].output)

    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = layers.Conv2D(512, (3, 3), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv5_1')(pool4)
    conv5_2 = layers.Conv2D(512, (3, 3), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv5_2')(conv5_1)
    conv5_3 = layers.Conv2D(512, (3, 3), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv5_3')(conv5_2)

    pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same', name='pool5')(conv5_3)

    fc6 = layers.Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation=activation, kernel_regularizer=l2(reg),
                        padding='same', kernel_initializer=initializer, name='fc6')(pool5)

    fc7 = layers.Conv2D(1024, (1, 1), activation=activation, padding='same', kernel_regularizer=l2(reg),
                        kernel_initializer=initializer, name='fc7')(fc6)

    conv8_1 = layers.Conv2D(256, (1, 1), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv8_1')(fc7)
    conv8_2 = layers.Conv2D(512, (3, 3), strides=(2, 2), activation=activation, kernel_regularizer=l2(reg),
                            padding='same', kernel_initializer=initializer, name='conv8_2')(conv8_1)

    conv9_1 = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv9_1')(conv8_2)
    conv9_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation=activation, padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer=initializer, name='conv9_2')(conv9_1)

    conv10_1 = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_regularizer=l2(reg),
                             kernel_initializer=initializer, name='conv10_1')(conv9_2)
    conv10_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation=activation, kernel_regularizer=l2(reg),
                             padding='valid', kernel_initializer=initializer, name='conv10_2')(conv10_1)

    conv11_1 = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_regularizer=l2(reg),
                             kernel_initializer=initializer, name='conv11_1')(conv10_2)
    conv11_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation=activation, kernel_regularizer=l2(reg),
                             padding='valid', kernel_initializer=initializer, name='conv11_2')(conv11_1)

    # defining locs
    loc_1 = layers.Conv2D(4 * 4, (1, 1), name='1st_bbs')(conv4_3)
    loc_2 = layers.Conv2D(6 * 4, (1, 1), name='2nd_bbs')(fc7)
    loc_3 = layers.Conv2D(6 * 4, (1, 1), name='3rd_bbs')(conv8_2)
    loc_4 = layers.Conv2D(6 * 4, (1, 1), name='4th_bbs')(conv9_2)
    loc_5 = layers.Conv2D(4 * 4, (1, 1), name='5th_bbs')(conv10_2)
    loc_6 = layers.Conv2D(4 * 4, (1, 1), name='6th_bbs')(conv11_2)

    # defining confidences
    conf_1 = layers.Conv2D(4 * 4, (1, 1), name='1st_conf')(conv4_3)
    conf_2 = layers.Conv2D(6 * 4, (1, 1), name='2nd_conf')(fc7)
    conf_3 = layers.Conv2D(6 * 4, (1, 1), name='3rd_conf')(conv8_2)
    conf_4 = layers.Conv2D(6 * 4, (1, 1), name='4th_conf')(conv9_2)
    conf_5 = layers.Conv2D(4 * 4, (1, 1), name='5th_conf')(conv10_2)
    conf_6 = layers.Conv2D(4 * 4, (1, 1), name='6th_conf')(conv11_2)

    # reshapes
    rloc_1 = layers.Reshape((-1, 4), name='rloc1')(loc_1)
    rloc_2 = layers.Reshape((-1, 4), name='rloc2')(loc_2)
    rloc_3 = layers.Reshape((-1, 4), name='rloc3')(loc_3)
    rloc_4 = layers.Reshape((-1, 4), name='rloc4')(loc_4)
    rloc_5 = layers.Reshape((-1, 4), name='rloc5')(loc_5)
    rloc_6 = layers.Reshape((-1, 4), name='rloc6')(loc_6)

    rconf_1 = layers.Reshape((-1, 4), name='rconf1')(conf_1)
    rconf_2 = layers.Reshape((-1, 4), name='rconf2')(conf_2)
    rconf_3 = layers.Reshape((-1, 4), name='rconf3')(conf_3)
    rconf_3 = layers.Activation('softmax')(rconf_3)
    rconf_4 = layers.Reshape((-1, 4), name='rconf4')(conf_4)
    rconf_4 = layers.Activation('softmax')(rconf_4)
    rconf_5 = layers.Reshape((-1, 4), name='rconf5')(conf_5)
    rconf_5 = layers.Activation('softmax')(rconf_5)
    rconf_6 = layers.Reshape((-1, 4), name='rconf6')(conf_6)
    rconf_6 = layers.Activation('softmax')(rconf_6)

    locs = layers.Concatenate(axis=1, name='all_bbox')([rloc_3, rloc_4, rloc_5, rloc_6])
    confs = layers.Concatenate(axis=1, name='all_preds')([rconf_3, rconf_4, rconf_5, rconf_6])

    final = layers.Concatenate(axis=2)([confs, locs])

    model = Model(inputs=BASE_MODEL.input, output=[final])

    for l in model.layers[:14]:
        l.trainable = True

    return model
