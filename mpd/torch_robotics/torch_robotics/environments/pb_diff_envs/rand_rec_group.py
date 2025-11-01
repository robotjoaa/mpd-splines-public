import matplotlib.pyplot as plt
import random
from typing import List
import numpy as np
from tqdm import tqdm
import os
import pdb
from abc import ABC, abstractmethod

from mpd.paths import DATASET_BASE_DIR
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.environments.dynamic_extension.sdf_utils import detect_primitive_overlaps
#from torch_robotics.environments.pb_diff_envs.utils.wall_primitives import RectangleWall
#from torch_robotics.environments.pb_diff_envs.utils.kuka_utils import from_rel2abs_path

class RandObjGroup(ABC) :
    def __init__(self, env_list_name:str, dataset_name:str, num_groups, num_walls, 
                 maze_size, half_extents, gap_betw_wall, seed, robot_class, 
                 gen_data=False, is_eval=False, tensor_args=DEFAULT_TENSOR_ARGS,
                 **kwargs) -> None:
        self.rng = np.random.default_rng(seed)
        self.rng_color = np.random.default_rng(100)
        self.env_list_name = env_list_name # mpd env name
        self.dataset_name = dataset_name # pbdiff env name 
        self.num_groups = num_groups

        self.maze_size = maze_size # env.limits_np

        self.gap_betw_wall = gap_betw_wall # prevent a separate divider
        self.sort_wloc = True # sort the order as other env does
        self.gen_data = gen_data # if False, directly load

        self.num_walls = num_walls # num_walls
        self.half_extents = np.copy(half_extents) # tuple of size 3? 2 -> np
        # assert half_extents.shape == (2,) and half_extents[0] == half_extents[1]

        self.wall_size = half_extents * 2 # tuple of size 3
        self.base_orn = [0, 0, 0, 1] # base orientation 

        #self.GUI = kwargs.get('GUI', False)
        #self.debug_mode = kwargs.get('debug_mode', False)
        # self.robot_class = robot_class
        # self.load_vis_objects = True # if load robot and table
        self.is_eval = is_eval
        self.rand_iter_limit = kwargs['rand_iter_limit'] # default 1e5
    
        self.tensor_args = tensor_args
        self.object_field = None
        
        if self.gen_data:
            self.create_envs()
        else:
            self.load_envs()

        self.compute_object_fields()


    def get_object_fields(self) : 
        return self.get_object_field

    @abstractmethod
    def compute_object_fields(self) : 
       raise NotImplementedError

    @abstractmethod 
    def create_envs(self):
        raise NotImplementedError

    @abstractmethod
    def load_envs(self) :
        raise NotImplementedError
  
class RandRectangleGroup(RandObjGroup):
    ''' 
    Migration of pb_diff_envs/rand_rec_group.py 
    Construct a group of obstacles in Rectangle Shape, each obstacle is represented in x-y location and half extent
    '''
    def __init__(self, env_list_name:str, dataset_name:str, num_groups, num_walls, 
                 maze_size, half_extents, gap_betw_wall, seed, robot_class, 
                 gen_data=False, is_eval=False, tensor_args=DEFAULT_TENSOR_ARGS,
                 **kwargs) -> None:
        
        super().__init__(env_list_name, dataset_name, num_groups, num_walls,
                         maze_size, half_extents, gap_betw_wall, seed, robot_class,
                         gen_data, is_eval, tensor_args,
                         **kwargs)


        
    def load_envs(self) : 
        self.rec_loc_grp_list = self.load_rec_loc_grp_list()
        rec_hExt_grp = np.stack([self.half_extents,] * self.num_walls, axis=0)
        self.rec_hExt_grp_list = rec_hExt_grp[None,].repeat(self.num_groups ,axis=0)
        # TODO : verify loaded rec_grp is not overlapping
        assert self.rec_hExt_grp_list.shape == (self.num_groups, self.num_walls, 2)
       

    def compute_object_fields(self) :
        self.object_fields = []
        for ng in range(self.num_groups) :
            # Convert half_extents to full sizes (MPD convention)
            new_field = MultiBoxField(centers = np.array([self.rec_loc_grp_list[ng]]),
                                      sizes = np.array([2 * self.rec_hExt_grp_list[ng]]),
                                      tensor_args=self.tensor_args)
            # ObjectField expects a list of primitives
            self.object_fields.append(ObjectField([new_field], f"object_{ng}"))
    
    def create_envs(self):
        '''set rec_loc_grp_list'''

        rec_loc_grp_list: list[np.ndarray] = [] # a list of np2d: n_c,3
        rec_hExt_grp_list = []
        for _ in tqdm(range(self.num_groups)):
            recloc_grp, rec_hExt_grp = self.sample_one_valid_env()
            rec_loc_grp_list.append(recloc_grp) # center location, np2d
            rec_hExt_grp_list.append(rec_hExt_grp) # np2d
        
        # np3d: ng, n_cube, 2
        self.rec_loc_grp_list = np.array(rec_loc_grp_list)
        self.rec_hExt_grp_list = np.array(rec_hExt_grp_list)

        self.save_rlg_list()

        return
    

    def sample_one_valid_env(self):
        '''create one env config without any overlap
        returns:
        array (n_c, 3) of 3d position '''
        rec_list = []
        cnt_iter = 0
        while len(rec_list) < self.num_walls:
            cnt_iter += 1
            if cnt_iter > self.rand_iter_limit: # deadlock, reset everything
                print('reach limit reset')
                for rec in rec_list: 
                    print(rec.center)
                rec_list = []
                cnt_iter = 0

            center_pos, hExt = self.sample_xy_hExt()
            tmp_rec = MultiBoxField(centers=np.array([center_pos]), 
                                    sizes=np.array([2*hExt]),
                                    tensor_args=self.tensor_args)
            tmp_list = [_ for _ in rec_list]
            tmp_list.append(tmp_rec)
            has_overlap, _ = detect_primitive_overlaps(tmp_list, margin=self.gap_betw_wall)

            if len(has_overlap) > 0 :
                pass
            else : 
                rec_list.append(tmp_rec)

        # print('cnt_iter', cnt_iter) # usually < 1000
        # self.reset()
        # pre sorted
        pos_list, hExt_list = self.get_pos_hExt_list(rec_list)
        # print('1 pos_list', pos_list)
        if self.sort_wloc:
            ## pos_list must be 2d, a list of list
            idx_and_pos = sorted(enumerate(pos_list), key=lambda i:i[1]) # list of a tuple (idx, w)
            pos_list = [i_p[1] for i_p in idx_and_pos] # i_p a tuple, still a list of list
            pos_sort_idx = [i_p[0] for i_p in idx_and_pos] # a list of int
            hExt_list = np.array(hExt_list)[pos_sort_idx] # switch order correspondingly
        else:
            assert False

        pos_list = np.array(pos_list) # n_c, 2
        # print('2 pos_list', pos_list)
        return pos_list, hExt_list
    

    # ------------- helpers for sampling ----------------
    def sample_xy_hExt(self):
        '''simply sample a location in all valid range, not consider overlap yet'''
        hExt = self.half_extents.copy()
        bottom_left, top_right = self.get_rec_center_range(hExt)
        while True:
            center_pos = self.rng.uniform(low=bottom_left, high=top_right) # (2,)
            # print('center_pos', center_pos)
            if False: # self.is_in_void_range(center_pos, half_extents=hExt):
                continue
            else:
                break
        return center_pos, hExt


    def get_rec_center_range(self, hExt):
        bottom_left = 0 + hExt + self.gap_betw_wall
        top_right = self.maze_size - hExt - self.gap_betw_wall
        return bottom_left, top_right
    
    # def check_is_overlap(self, rec_list, rec_new):
    #     for rec in rec_list:
    #         #is_ovlp = RectangleWall.is_recWall_overlap(rec, rec_new, self.gap_betw_wall)
    #         is_ovlp = check_box_box_overlap(rec.center, rec.hExt, 
    #                                         rec_new.center, rec_new.hExt, 
    #                                         margin = self.gap_betw_wall)
    #         if is_ovlp: # overlap
    #             return True
    #     return False

    def get_pos_hExt_list(self, rec_list): # List[MultiBoxField(single primitives)]
        '''from a list of rectangle to a list of list'''
        pos_list = []
        hExt_list = []
        for rec in rec_list:
            pos_list.append(rec.centers.squeeze().tolist())
            hExt_list.append(rec.half_sizes.squeeze().tolist())
        return pos_list, hExt_list

    # --------- save and load the wall locations -----------
    def get_npyname(self):
        self.prefix = os.path.join(DATASET_BASE_DIR, self.env_list_name, 'rand_rec2dgrp')
        os.makedirs(self.prefix, exist_ok=True)
        if self.is_eval:
            npyname = f'{self.prefix}/{self.dataset_name}_eval.npy'
        else:
            npyname = f'{self.prefix}/{self.dataset_name}.npy'
        return npyname
    
    ## np (num_groups,3)
    def save_rlg_list(self):
        assert self.rec_loc_grp_list.shape[0] == self.num_groups
        npyname = self.get_npyname()
        ## check instead
        if os.path.exists(npyname) and 'testOnly' not in npyname:
            # assert False
            self.check_matched()
        else:
            np.save(npyname, self.rec_loc_grp_list)
            if 'testOnly' not in npyname:
                os.chmod(npyname, 0o444)

    def load_rec_loc_grp_list(self):
        rec_loc_grp_list = np.load(self.get_npyname())
        return rec_loc_grp_list


    def check_matched(self):
        assert (self.rec_loc_grp_list == self.load_rec_loc_grp_list()).all()