import omni.isaac.core.utils.bounds as bounds_utils
import torch
import omni.isaac.core.utils.prims as prim_utils


def GetPrimsBoundingBox(prims, stage):
    bb_cache = bounds_utils.create_bbox_cache()
    bbs = torch.zeros((len(prims), 6), device='cuda:0')
    for i, prim in enumerate(prims):
        bb = bounds_utils.compute_aabb(bb_cache, prim_path=prim, include_children=True)
        bbs[i] = torch.tensor(bb, device='cuda:0')
    return bbs


def GetPrimsPathWithMatchingExpression(expression):
    return prim_utils.find_matching_prim_paths(expression)
