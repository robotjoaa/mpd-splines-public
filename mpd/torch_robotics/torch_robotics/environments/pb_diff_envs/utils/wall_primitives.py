from typing import List
import numpy as np

'''
Should be replaced by Multi[Object]Field
'''
# class RectangleWall:
#     '''a single rectangle, like one wall'''
#     def __init__(self, center: np.ndarray, hExt: np.ndarray) -> None:
#         self.center = np.copy(center)
#         self.hExt = np.copy(hExt)
#         self.top_right = center + hExt
#         self.bottom_left = center - hExt

#     def update_center_pos(self, center):
#         '''used for dynamic wall'''
#         self.center = center
#         self.top_right = center + self.hExt
#         self.bottom_left = center - self.hExt
#         # pdb.set_trace()

#     def set_config(self, center):
#         assert type(center) == np.ndarray and center.shape == (2,)
#         self.update_center_pos(center)

#     def is_point_inside(self, pose:np.ndarray, min_to_wall_dist:float):
#         '''True if collision, input is unnormed'''
#         # pose (2,)
#         cond_1 = ( pose > ( self.bottom_left -  min_to_wall_dist) ).all()
#         cond_2 = ( pose < ( self.top_right + min_to_wall_dist) ).all()
#         return cond_1 & cond_2
    # should use detect_primitive_overlap
    # @classmethod
    # def is_recWall_overlap(rec_1, rec_new, gap):
    #     ''' 
    #     True if there is overlap
    #     make sure that rec_1 is the one in the maze,
    #     rec_2 is the new one to check,
    #     we enlarge rec_1 a little bit
    #     '''

    #     assert type(rec_1) == RectangleWall and type(rec_new) == RectangleWall

    #     return not (rec_1.top_right[0] + gap  < rec_new.bottom_left[0]
    #         or rec_1.bottom_left[0] - gap > rec_new.top_right[0]
    #         or rec_1.top_right[1] + gap   < rec_new.bottom_left[1]
    #         or rec_1.bottom_left[1] - gap > rec_new.top_right[1])


'''
Should be replaced by Multi[Object]Field
'''
# class RectangleWallGroup:
#     '''
#     a group of rectangles, typically as *one* env
#     '''
#     def __init__(self, recWall_list: List[RectangleWall]) -> None:
#         self.recWall_list = recWall_list
    
#     def is_point_inside_wg(self, pose, min_to_wall_dist):
#         '''min_to_wall_dist: if 0.01, cannot be in region nearer than 0.01'''
#         for recWall in self.recWall_list:
#             is_col = recWall.is_point_inside(pose, min_to_wall_dist)
#             if is_col:
#                 return True
#         return False





