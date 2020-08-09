import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]
#----------------------------------------------------------------------------
# Generator loss function.

def G_wgan_acgan(G, D, lod, labels, well_facies, prob_images, minibatch_size,
    Wellfaciesloss_weight = 0.25, MudProp_weight = 0.2, Width_weight = 0.2, Sinuosity_weight = 0.2, orig_weight = 1, labeltypes = None, Probimageloss_weight = 0.2, batch_multiplier = 8, lossnorm = False): 
#labeltypes, e.g., labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity', set in config file
    # loss for channel orientation is not designed below, so do not include "0" in labeltypes.
    # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.
    
    label_size = len(labeltypes)
    if label_size == 0: 
        labels_in = labels
    else:     
        labels_list = []
        for k in range(label_size):
            labels_list.append(tf.random.uniform(([minibatch_size]), minval=-1, maxval=1))
        if 1 in labeltypes:   # mud proportion
            ind = labeltypes.index(1)
            labels_list[ind] = tf.clip_by_value(labels[:, ind] + tf.random.uniform([minibatch_size], minval=-0.2, maxval=0.2), -1, 1)    
        labels_in = tf.stack(labels_list, axis = 1)          
    
    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
    labels_lg = tf.reshape(tf.tile(tf.expand_dims(labels_in, 1), [1, batch_multiplier, 1]), ([-1] + G.input_shapes[1][1:])) 
    well_facies = tf.cast(well_facies, tf.float32)
    well_facies_lg = tf.reshape(tf.tile(tf.expand_dims(well_facies, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))
    prob_images = tf.cast(prob_images, tf.float32)
    prob_images_lg = tf.reshape(tf.tile(tf.expand_dims(prob_images, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
    fake_images_out = G.get_output_for(latents, labels_lg, well_facies_lg, prob_images_lg, is_training=True)  
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    if lossnorm: loss = (loss -14.6829250772099) / 4.83122039859412   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     

    with tf.name_scope('LabelPenalty'):
        def addMudPropPenalty(index):
            MudPropPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: MudPropPenalty = (MudPropPenalty -0.36079434843794) / 0.11613414177144  # To normalize this loss 
            MudPropPenalty = tfutil.autosummary('Loss_G/MudPropPenalty', MudPropPenalty)        
            MudPropPenalty = MudPropPenalty * MudProp_weight  
            return loss+MudPropPenalty
        if 1 in labeltypes:
            ind = labeltypes.index(1)
            loss = addMudPropPenalty(ind)
            
        def addWidthPenalty(index):
            WidthPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: WidthPenalty = (WidthPenalty -0.600282781464712) / 0.270670509379704  # To normalize this loss 
            WidthPenalty = tfutil.autosummary('Loss_G/WidthPenalty', WidthPenalty)             
            WidthPenalty = WidthPenalty * Width_weight            
            return loss+WidthPenalty
        if 2 in labeltypes:
            ind = labeltypes.index(2)
            loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addWidthPenalty(ind), lambda: loss)
        
        def addSinuosityPenalty(index):
            SinuosityPenalty = tf.nn.l2_loss(labels_lg[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: SinuosityPenalty = (SinuosityPenalty -0.451279248935835) / 0.145642580091667  # To normalize this loss 
            SinuosityPenalty = tfutil.autosummary('Loss_G/SinuosityPenalty', SinuosityPenalty)            
            SinuosityPenalty = SinuosityPenalty * Sinuosity_weight              
            return loss+SinuosityPenalty
        if 3 in labeltypes:
            ind = labeltypes.index(3)
            loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addSinuosityPenalty(ind), lambda: loss)
 

    def Wellpoints_L2loss(well_facies, fake_images):
        loss = tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:2] - tf.where((fake_images+1)/2>0.4, tf.fill(tf.shape(fake_images), 1.), (fake_images+1)/2)))
        loss = loss / tf.reduce_sum(well_facies[:, 0:1])
        return loss
    def addwellfaciespenalty(well_facies, fake_images_out, loss, Wellfaciesloss_weight):
        with tf.name_scope('WellfaciesPenalty'):
            WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_images_out)   # as levee is 0.5, in well_facies data, levee and channels' codes are all 1        
            if lossnorm: WellfaciesPenalty = (WellfaciesPenalty -0.00887323171768953) / 0.00517647244943928
            WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
            loss += WellfaciesPenalty * Wellfaciesloss_weight   
        return loss   
    loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 3.)), lambda: addwellfaciespenalty(well_facies_lg, fake_images_out, loss, Wellfaciesloss_weight), lambda: loss)
       
        
    def addfaciescodedistributionloss(probs, fakes, weight, batchsize, relzs, loss):  # used when resolution is 64x64        
        with tf.name_scope('ProbimagePenalty'):   
            # In paper, only probability map for channel complex is condisered. If multiple probability maps for multiple facies are considered, needs to calculate channelindicator and probPenalty for each facies.          
            channelindicator = 1 / (1+tf.math.exp(-16*(fakes+0.5))) # use adjusted sigmoid to replace thresholding.  
            probs_fake = tf.reduce_mean(tf.reshape(channelindicator, ([batchsize, relzs] + G.input_shapes[3][1:])), 1)            
            #****** L2 loss
            ProbPenalty = tf.nn.l2_loss(probs - probs_fake)  # L2 loss
            if lossnorm: ProbPenalty = ((ProbPenalty*tf.cast(relzs, tf.float32))-19134)/5402   # normalize
            ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
        loss += ProbPenalty * weight
        return loss
 
    loss = addfaciescodedistributionloss(prob_images, fake_images_out, Probimageloss_weight, minibatch_size, batch_multiplier, loss)      
     
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)    
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, opt, minibatch_size, reals, labels, well_facies, prob_images,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    label_weight    = 10,       # Weight of the conditioning terms.
    batch_multiplier = 1):       

    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
    prob_images = tf.reshape(tf.tile(tf.expand_dims(prob_images, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
    well_facies = tf.reshape(tf.tile(tf.expand_dims(well_facies, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))
    labels = tf.reshape(tf.tile(tf.expand_dims(labels, 1), [1, batch_multiplier, 1]), ([-1] + G.input_shapes[1][1:]))
    
    fake_images_out = G.get_output_for(latents, labels, well_facies, prob_images, is_training=True)
    
    reals = tf.reshape(tf.tile(tf.expand_dims(reals, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
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
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon

    with tf.name_scope('LabelPenalty'):
        label_penalty_reals = tf.nn.l2_loss(labels - real_labels_out)                            
        label_penalty_fakes = tf.nn.l2_loss(labels - fake_labels_out)
        label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)
        label_penalty_fakes = tfutil.autosummary('Loss_D/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * label_weight
        loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss