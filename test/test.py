from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from mpd.parametric_trajectory.trajectory_bspline import BSpline
#from torch_robotics.torch_utils.torch_utils import to_torch, dict_to_device, to_numpy, DEFAULT_TENSOR_ARGS

import matplotlib.pyplot as plt
import numpy as np

def test() : 
    n_control_points = 18
    degree = 5
    num_T_pts = 128
    #tensor_args = DEFAULT_TENSOR_ARGS

    spline = BSpline(num_pts=n_control_points, degree=degree, num_T_pts=num_T_pts)

    print("N shape : ",spline.N.shape)
    print("dN shape : ",spline.dN.shape)
    print("ddN shape : ",spline.ddN.shape)

    plot_basis(spline)


def plot_basis(spline) : 

    fig, axs = plt.subplots(3,1,figsize=(9, 5))
    fig.tight_layout()

    Ns = spline.N 
    dNs = spline.dN
    ddNs = spline.ddN
    num_T_pts = Ns.shape[1]
    T = np.linspace(0.0, 1.0, num_T_pts)
    i = Ns.shape[2]-1
    axs[0].plot(T, Ns[0,:,])
    axs[0].set_title(f'N{i},{spline.d}')
    #for i in range(b_mat.shape[0]):
    axs[1].plot(T, dNs[0,:,:])
    axs[1].set_title(f'dN{i},{spline.d}')
    axs[2].plot(T, ddNs[0,:,:])
    axs[2].set_title(f'ddN{i},{spline.d}')
    # plt.title(f'B-spline Basis Functions (degree={spline.d}, n={spline.n_pts})')
    # plt.xlabel('t (normalized parameter)')
    # plt.ylabel('Basis Function Value')

    # #plt.legend()
    # plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test()
