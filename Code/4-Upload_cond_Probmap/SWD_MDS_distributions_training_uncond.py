
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from sklearn import manifold
from matplotlib import pyplot as plt
from scipy.stats import kde


import scipy.ndimage

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


#----------------------------------------------------------------------------
# EDIT: added

class API:
    def __init__(self, image_shape, image_dtype, num_images_per_group, num_groups_test, num_groups_fake):
        self.nhood_size         = 5
        self.nhoods_per_image   = 24
        self.dir_repeats        = 4
        self.dirs_per_repeat    = 64
        self.resolutions = []
        res = image_shape[1]
        self.num_groups_test = num_groups_test
        self.num_groups_fake = num_groups_fake
        self.num_groups_total = self.num_groups_test + self.num_groups_fake
        self.num_images_per_group = num_images_per_group
        
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        descriptors = [[] for res in self.resolutions]

    def feed(self, mode, images, result_subdir):
        groups_lap = []
        for i in range(self.num_groups_total):
            minibatch = images[i * self.num_images_per_group : (i + 1) * self.num_images_per_group]
            descriptors = [[] for res in self.resolutions]
            for lod, level in enumerate(generate_laplacian_pyramid(minibatch, len(self.resolutions))):
                desc = get_descriptors_for_minibatch(level, self.nhood_size, self.nhoods_per_image)
                descriptors[lod].append(desc)
            groups_lap.append(descriptors)
 
        kk = np.tril(np.ones((self.num_groups_total,self.num_groups_total)), -1)  
        coor = np.argwhere(kk > 0)
        list_1 = coor[:, 0]
        list_2 = coor[:, 1]  
        
        gr_swd = []
        for gr in range(list_1.shape[0]):
            desc_1 = [finalize_descriptors(d) for d in groups_lap[list_1[gr]]]
            desc_2 = [finalize_descriptors(d) for d in groups_lap[list_2[gr]]]
            slw = [sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat) for dreal, dfake in zip(desc_1, desc_2)]
            #gr_swd.append(np.mean(slw) * 1e3)
            gr_swd.append((slw[0]*0.2+slw[1]*0.3+slw[2]*0.5) * 1e3)
        
        def convert_to_matrix(a):
            n = int(np.sqrt(len(a)*2))+1
            mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
            out = np.zeros((n,n),dtype=float)
            out[mask] = a
            np.transpose(out)[mask] = a        
            return out        

        swd_matrix = convert_to_matrix(gr_swd)
        
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
        coos = mds.fit(swd_matrix).embedding_
        
        x_real_co = coos[:self.num_groups_test, 0]
        y_real_co = coos[:self.num_groups_test, 1]
        x_fake_trad_co = coos[self.num_groups_test:self.num_groups_test+ self.num_groups_fake, 0]
        y_fake_trad_co = coos[self.num_groups_test:self.num_groups_test+ self.num_groups_fake, 1]  
        
        plot_lim_min = -700
        plot_lim_max = 700
        # Create a figure with 6 plot areas
        fig, axes = plt.subplots(ncols=2, nrows=1, sharey='row')
        fig.set_size_inches(10, 4, forward=True)
        
        axes[0].set_title('Scatterplot')
        axes[0].set_xlim([plot_lim_min, plot_lim_max])
        axes[0].set_ylim([plot_lim_min, plot_lim_max])        
        axes[0].plot(x_real_co, y_real_co, 'ro', label = 'Real')
        axes[0].plot(x_fake_trad_co, y_fake_trad_co, 'b+', label = 'Generated')
        axes[0].legend(loc='upper right')
        
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins = 40
        k_real = kde.gaussian_kde((coos.T[:, :self.num_groups_test]))
        xi_real, yi_real = np.mgrid[plot_lim_min:plot_lim_max:nbins*1j, plot_lim_min:plot_lim_max:nbins*1j]
        zi_real = k_real(np.vstack([xi_real.flatten(), yi_real.flatten()]))
        axes[1].set_xlim([plot_lim_min, plot_lim_max])
        axes[1].set_ylim([plot_lim_min, plot_lim_max])                
        real_contr = axes[1].contour(xi_real, yi_real, zi_real.reshape(xi_real.shape), 6, colors='r', label = 'Real') 

        k_fake_trad = kde.gaussian_kde((coos.T[:, self.num_groups_test:self.num_groups_test+self.num_groups_fake]))
        xi_fake_trad, yi_fake_trad = np.mgrid[plot_lim_min:plot_lim_max:nbins*1j, plot_lim_min:plot_lim_max:nbins*1j]
        zi_fake_trad = k_fake_trad(np.vstack([xi_fake_trad.flatten(), yi_fake_trad.flatten()]))
        fake_contr_trad = axes[1].contour(xi_fake_trad, yi_fake_trad, zi_fake_trad.reshape(xi_fake_trad.shape), 5, colors='blue',  linestyles= 'dashed', label = 'Generated') 
        
     #   k_fake_prog = kde.gaussian_kde((coos.T[:, num_groups*2:]))
    #    xi_fake_prog, yi_fake_prog = np.mgrid[plot_lim_min:plot_lim_max:nbins*1j, plot_lim_min:plot_lim_max:nbins*1j]
    #    zi_fake_prog = k_fake_prog(np.vstack([xi_fake_prog.flatten(), yi_fake_prog.flatten()]))
    #    fake_contr_prog = axes[1].contour(xi_fake_prog, yi_fake_prog, zi_fake_prog.reshape(xi_fake_prog.shape), 5, colors='k', linestyles ='dashdot', label = 'GroundTruthFM') 
        
        axes[1].set_title('Densityplot')
        
        real_contr.collections[0].set_label('Real')
        fake_contr_trad.collections[0].set_label('Generated')
     #   fake_contr_prog.collections[0].set_label('GroundTruthFM')
        axes[1].legend(loc='upper right') 
        
       # plt.savefig(result_subdir + '/SWD_MDS distribution_test_generated_groundtruth.jpg' , dpi=200)


#----------------------------------------------------------------------------
