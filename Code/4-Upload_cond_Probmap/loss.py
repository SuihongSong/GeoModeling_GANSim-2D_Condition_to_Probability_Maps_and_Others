import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]
#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, lod, training_set, labels, prob_images, minibatch_size,
    Probimageloss_weight = 0.1, orig_weight = 1.0, batch_multiplier = 8, lossnorm = True): 

    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
    prob_images = tf.cast(prob_images, tf.float32)
    prob_images_lg = tf.reshape(tf.tile(tf.expand_dims(prob_images, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))   
    
    labels = tf.zeros([minibatch_size * batch_multiplier] + G.input_shapes[1][1:])  # ONly used for test of the new loss of probmaps
    
    fake_images_out = G.get_output_for(latents, labels, prob_images_lg, is_training=True)  
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    loss = loss / batch_multiplier
    if lossnorm: loss = (loss -14.6829250772099) / 4.83122039859412   #To Normalize    
    loss = tfutil.autosummary('Loss_G/WGAN_loss', loss)
    loss = orig_weight * loss          

    def addfaciescodedistributionloss(probs, fakes, weight, batchsize, relzs, loss):  # used when resolution is 64x64        
        with tf.name_scope('ProbimagePenalty'):            
            # In paper, only probability map for channel complex is condisered. If multiple probability maps for multiple facies are considered, needs to calculate channelindicator and probPenalty for each facies.  
            channelindicator = 1 / (1+tf.math.exp(-16*(fakes+0.5))) # use adjusted sigmoid function as an continous indicatorization.       
            probs_fake = tf.reduce_mean(tf.reshape(channelindicator, ([batchsize, relzs] + G.input_shapes[2][1:])), 1)             
            ProbPenalty = tf.nn.l2_loss(probs - probs_fake)  # L2 loss
            if lossnorm: ProbPenalty = ((ProbPenalty*tf.cast(relzs, tf.float32))-19134)/5402   # normalize
            ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
        loss += ProbPenalty * weight
        return loss 
    loss = addfaciescodedistributionloss(prob_images, fake_images_out, Probimageloss_weight, minibatch_size, batch_multiplier, loss)     
 
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, prob_images,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 10,       # Weight of the conditioning terms.
    batch_multiplier = 1):       

    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
    prob_images = tf.reshape(tf.tile(tf.expand_dims(prob_images, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))   
    
    labels = tf.zeros([minibatch_size * batch_multiplier] + G.input_shapes[1][1:])  # ONly used for test of the new loss of probmaps
    
    fake_images_out = G.get_output_for(latents, labels, prob_images, is_training=True)
    
    reals = tf.reshape(tf.tile(tf.expand_dims(reals, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))   
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size * batch_multiplier, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        #mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    
    loss = loss / batch_multiplier
    return loss

#----------------------------------------------------------------------------
