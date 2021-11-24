# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:55:39 2021

@author: nsenanay
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate



def conv_block(input, num_filters):
    x= Conv2D(num_filters,3, padding="same")(input)
    x=BatchNormalization()(x)
    x=Activation('relu')
    
    x= Conv2D(num_filters,3, padding='same')
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    return x


def encoder_block(input, num_filters):
    x= conv_block(input, num_filters)
    p=MaxPool2D(pool_size=(2,2))(x)
    return x,p

def decorder_block(input, skip_features, num_filters):
    x=Conv2DTranspose(num_filters,2,strides=2,padding='same')(input)
    x=Concatenate()[x,skip_features]
    x= conv_block(x,num_filters)
    return x

def build_u_net(input_shape):
    input= Input(input_shape)
    s1,p1= encoder_block(input, 64)
    s2,p2=encoder_block(p1, 128)
    s3,p3= encoder_block(p2,256)
    s4,p4= encoder_block(p3, 512)
    
    b1= conv_block(p4, 1024)
    
    d1= decorder_block(b1, s4, 512)
    d2=decorder_block(d1,s3,256)
    d3= decorder_block(d2, s2, 128)
    d4 = decorder_block(d3, s1, 64)
    out_put = Conv2D(1,1,padding='same', activation="sigmoid") (d4)
    model=Model(input, out_put,name='u-net')
    return model
    