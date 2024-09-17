#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:02:13 2024

@author: ivo
"""
print('TEST')

import sys
import dapper.mods as modelling

import keras
import tensorflow as tf


for a in sys.argv:
    print(tf.config.experimental.list_physical_devices())
    print('Arg ',a)
