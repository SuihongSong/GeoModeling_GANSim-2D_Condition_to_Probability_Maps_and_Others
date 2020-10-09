# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = '/scratch/users/suihong/DataSets(MultiChannels_V4_Consistency_HighestFrequcyDownSampling)/'
# "data_dir" refers to the path of grandparent directory of training dataset like *.tfrecord files. "dataset" in line 33 refers to parent folder name of training dataset.
# e.g., folder "AA/BB/CC" includes all the *.tfrecord files training dataset, then data_dir = 'AA/BB/', and in line 33, tfrecord_dir=  'CC'. 

result_dir = '/scratch/users/suihong/ProGAN_MultiChannel_Reusults_ConditionedtoMultiConditions_TF'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict(tfrecord_dir='TrainingData')         # Replace 'TrainingData' with parent folder name of *.tfrecords training dataset.  
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='6by8', layout='random')       # Options for train.setup_snapshot_image_grid().

desc += '-CondProb';  

desc += '-2gpu'; num_gpus = 2; sched.minibatch_base = 32; sched.minibatch_dict = {4: 32, 8: 32, 16: 32, 32: 32, 64: 32}; sched.G_lrate_dict = {4: 0.0025, 8: 0.005, 16: 0.005, 32: 0.0035, 64: 0.0025}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
train.total_kimg = 60000
sched.max_minibatch_per_gpu = {32: 32, 64: 32}
