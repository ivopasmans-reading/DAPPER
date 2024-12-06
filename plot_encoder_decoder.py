#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:19:07 2024

@author: ivo
"""

from graphviz import Digraph


class NeuralNetork: 
    
    def __init__(self):
        self.graph = Digraph(format='png')
        self.graph.attr(rankdir='LR')
        
    def add_layer(self, layer, **kwargs):
        with self.graph.subgraph(name=layer.name) as c:
            c.attr(label=layer.name, **kwargs)
            for name, label in layer.nodes:
                c.node(name, label)
                
    def fully_connect(self, layer_left, layer_right):
        for left,_ in layer_left.nodes:
            for right,_ in layer_right.nodes:
                self.graph.edge(left,right, arrowhead='none')
                
    def render(self, name):
        self.graph.render(name, view=True)
    
class Layer:
    
    def __init__(self, name, labels):
        self.name = name
        self.labels = labels 
        self.names = [f"{name}_{n:03d}" for n,_ in enumerate(self.labels)]
        self.nodes = [(name,label) for name,label in zip(self.names, self.labels)]


#%% Encoder

inputs = [Layer('input',['x0','x1'])]
mean  = [Layer(f"hidden_zmean_{layer:d}",[""]*8) for layer in range(6)]
mean += [Layer("recenter",["-"]), Layer("z_mean",["z_mean"])]
var   = [Layer(f"hidden_zvar_{layer:d}",[""]*8) for layer in range(6)]
var  += [Layer("rescale",["/"]), Layer("z_var",["log_z_var"])]


network = NeuralNetork()

network.add_layer(inputs[0])
for left, right in zip(inputs+mean[:-1], mean):
    network.add_layer(right)
    network.fully_connect(left, right)
for left, right in zip(inputs+var[:-1], var):
    network.add_layer(right)
    network.fully_connect(left, right)
    
network.render("encoder")


#%% Decoder

inputs = [Layer('input',['z']), Layer('rescale',['*'])]
mean  = [Layer(f"hidden_xmean_{layer:d}",[""]*8) for layer in range(6)]
mean += [Layer("x_mean",["x_mean"])]
var   = [Layer(f"hidden_xvar_{layer:d}",[""]*8) for layer in range(6)]
var  += [Layer("x_var",["log_x_var"])]

network = NeuralNetork()

network.add_layer(inputs[0])
network.add_layer(inputs[1])
network.fully_connect(inputs[0], inputs[1])
for left, right in zip(inputs[1:]+mean[:-1], mean):
    network.add_layer(right)
    network.fully_connect(left, right)
for left, right in zip(inputs[1:]+var[:-1], var):
    network.add_layer(right)
    network.fully_connect(left, right)
    
network.render("decoder")
