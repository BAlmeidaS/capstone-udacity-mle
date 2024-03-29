from keras.applications import VGG19, VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input as preprocess_19
from keras.applications.vgg16 import preprocess_input as preprocess_16
from keras.models import Model

from keras.regularizers import l2

from keras import backend as K
from keras import layers

VGG = VGG19(weights='imagenet', include_top=False, input_shape=(624, 624, 3))
BASE_MODEL = Model(inputs=VGG.input, outputs=VGG.get_layer('block5_conv4').output)


def ssd_model_624_vgg_19(reg=0.0003):
    pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                padding='same', name='pool5')(BASE_MODEL.output)

    fc6 = layers.Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', kernel_regularizer=l2(reg),
                        padding='same', kernel_initializer='he_normal', name='fc6')(pool5)

    fc7 = layers.Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                        kernel_initializer='he_normal', name='fc7')(fc6)

    conv8_1 = layers.Conv2D(256, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer='he_normal', name='conv8_1')(fc7)
    conv8_2 = layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', kernel_regularizer=l2(reg),
                            padding='valid', kernel_initializer='he_normal', name='conv8_2')(conv8_1)

    conv9_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer='he_normal', name='conv9_1')(conv8_2)
    conv9_2 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_regularizer=l2(reg),
                            kernel_initializer='he_normal', name='conv9_2')(conv9_1)

    conv10_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer='he_normal', name='conv10_1')(conv9_2)
    conv10_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(reg),
                            padding='valid', kernel_initializer='he_normal', name='conv10_2')(conv10_1)

    conv11_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                            kernel_initializer='he_normal', name='conv11_1')(conv10_2)
    conv11_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(reg),
                            padding='valid', kernel_initializer='he_normal', name='conv11_2')(conv11_1)

    conv12_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                             kernel_initializer='he_normal', name='conv12_1')(conv11_2)
    conv12_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(reg),
                             padding='valid', kernel_initializer='he_normal', name='conv12_2')(conv12_1)

    conv13_1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(reg),
                             kernel_initializer='he_normal', name='conv13_1')(conv12_2)
    conv13_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(reg),
                             padding='valid', kernel_initializer='he_normal', name='conv13_2')(conv13_1)

    final = conv13_2

    # defining locs
    loc_1 = layers.Conv2D(3 * 4, (3, 3), padding='same', name='1st_bbs')(BASE_MODEL.layers[-6].output)
    loc_2 = layers.Conv2D(3 * 4, (3, 3), padding='same', name='2nd_bbs')(fc7)
    loc_3 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='3rd_bbs')(conv8_2)
    loc_4 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='4th_bbs')(conv9_2)
    loc_5 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='5th_bbs')(conv10_2)
    loc_6 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='6th_bbs')(conv11_2)
    loc_7 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='7th_bbs')(conv12_2)
    loc_8 = layers.Conv2D(5 * 4, (3, 3), padding='same', name='8th_bbs')(conv13_2)

    # defining confidences
    conf_1 = layers.Conv2D(3 * 2, (3, 3), padding='same', name='1st_conf')(BASE_MODEL.layers[-6].output)
    conf_2 = layers.Conv2D(3 * 2, (3, 3), padding='same', name='2nd_conf')(fc7)
    conf_3 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='3rd_conf')(conv8_2)
    conf_4 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='4th_conf')(conv9_2)
    conf_5 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='5th_conf')(conv10_2)
    conf_6 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='6th_conf')(conv11_2)
    conf_7 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='7th_conf')(conv12_2)
    conf_8 = layers.Conv2D(5 * 2, (3, 3), padding='same', name='8th_conf')(conv13_2)

    # reshapes
    rloc_1 = layers.Reshape((-1, 4), name='rloc1')(loc_1)
    rloc_2 = layers.Reshape((-1, 4), name='rloc2')(loc_2)
    rloc_3 = layers.Reshape((-1, 4), name='rloc3')(loc_3)
    rloc_4 = layers.Reshape((-1, 4), name='rloc4')(loc_4)
    rloc_5 = layers.Reshape((-1, 4), name='rloc5')(loc_5)
    rloc_6 = layers.Reshape((-1, 4), name='rloc6')(loc_6)
    rloc_7 = layers.Reshape((-1, 4), name='rloc7')(loc_7)
    rloc_8 = layers.Reshape((-1, 4), name='rloc8')(loc_8)

    rconf_1 = layers.Reshape((-1, 2), name='rconf1')(conf_1)
    rconf_2 = layers.Reshape((-1, 2), name='rconf2')(conf_2)
    rconf_3 = layers.Reshape((-1, 2), name='rconf3')(conf_3)
    rconf_4 = layers.Reshape((-1, 2), name='rconf4')(conf_4)
    rconf_5 = layers.Reshape((-1, 2), name='rconf5')(conf_5)
    rconf_6 = layers.Reshape((-1, 2), name='rconf6')(conf_6)
    rconf_7 = layers.Reshape((-1, 2), name='rconf7')(conf_7)
    rconf_8 = layers.Reshape((-1, 2), name='rconf8')(conf_8)

    confs = layers.Concatenate(axis=1, name='all_preds')([rconf_1, rconf_2, rconf_3, rconf_4,
                                                          rconf_5, rconf_6,
                                                          rconf_7, rconf_8
                                                          ])
    confs_softmax = layers.Activation('softmax')(confs)

    locs = layers.Concatenate(axis=1, name='all_bbox')([rloc_1, rloc_2, rloc_3, rloc_4,
                                                        rloc_5, rloc_6,
                                                        rloc_7, rloc_8
                                                        ])

    final = layers.Concatenate(axis=2)([confs_softmax, locs])

    model = Model(inputs=BASE_MODEL.input, output=[final])

   # for l in model.layers[:6]:
   #     l.trainable = False

    return model
