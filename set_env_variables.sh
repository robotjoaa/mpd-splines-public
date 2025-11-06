# export LD_LIBRARY_PATH=$HOME/miniconda3/envs/mpd-splines-public/lib
# export CPATH=$HOME/miniconda3/envs/mpd-splines-public/include
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/mpd-splines-public/lib:$LD_LIBRARY_PATH"
export CPATH=$HOME/anaconda3/envs/mpd-splines-public/include

# unset LD_PRELOAD to use conda libstdc
unset LD_PRELOAD