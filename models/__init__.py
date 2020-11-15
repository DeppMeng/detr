# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_vis import build_vis


def build_model(args):
    return build(args)

def build_vis_model(args):
    return build_vis(args)
