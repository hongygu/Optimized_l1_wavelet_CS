import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import dwtcoeffs
import nodes

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
nb_blocks = 10
epochs = 100
batchSize = 1
acc_rate = 4
# Parameters
num_gpus = [1]
learning_rate = 5e-3
level = 4
subbnad_num = 14
wavenum = 4
eps = 1e-9
wavelet_type1 = dwtcoeffs.db4
wavelet_type2 = dwtcoeffs.db3
wavelet_type3 = dwtcoeffs.db2
wavelet_type4 = dwtcoeffs.haar
data_opt = 'Coronal-PD'
LR = '5e3_'

data_tag = '4wavelet_reweight'
# user should specify the directories for data, coil, mask and reweight directory
data_dir = "..."
coil_dir = "..."
mask_dir = '...'
reweight_dir = '....mat' #the directory of the first set of reconstructed images
nrow_GLOB, ncol_GLOB, ncoil_GLOB = 320, 368, 15 #change the dimensions of the dataset if necessary


# function c2r_tf contatenate complex input as new axis two two real input,axis=-1-->mxn-->mxnx2
c2r_tf = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)
# r2c_tf takes the last dimension of real input and converts to complex
r2c_tf = lambda x: tf.complex(x[..., 0], x[..., 1])

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype == 'complex64':
        dtype = np.float32
    else:
        dtype = np.float64
    out = np.zeros(inp.shape + (2,), dtype=dtype)
    out[..., 0] = inp.real
    out[..., 1] = inp.imag
    return out


def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128
    out = np.zeros(inp.shape[0:2], dtype=dtype)
    out = inp[..., 0] + 1j * inp[..., 1]
    return out


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:
        fact = 1
        for axis in axes:
            fact = fact * kspace.shape[axis]
        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:
        fact = 1
        for axis in axes:
            fact = fact * ispace.shape[axis]
        ispace = ispace * np.sqrt(fact)

    return ispace


def tf_flip2D(input_data, axes=1):
    nx = int(nrow_GLOB / 2)
    ny = int(ncol_GLOB / 2)
    nc = ncoil_GLOB
    if axes == 1:
        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])
    elif axes == 2:
        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])
    else:
        raise ValueError('Invalid nums of dims')
    return tf.concat([second_half, first_half], axis=axes)

def tf_fftshift(input_x, axes=1):
    return tf_flip2D(tf_flip2D(input_x, axes=1), axes=2)

# Build the function to average the gradients
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


def getLambda():
    """
    create a shared variable (across unrolled iterations) called lambda.
    """
    with tf.variable_scope('Lambda'):
        lam = tf.get_variable(name='lam', shape=(wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(lam)

def getEtta():
    """
    create a shared variable (across unrolled iterations) called etta.
    """
    with tf.variable_scope('Etta'):
        etta = tf.get_variable(name='etta', shape=(wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(etta)

def getScaling_factor():
    """
    create a shared variable (across unrolled iterations) called scaling_factor.
    """
    with tf.variable_scope('Scaling_factor'):
        scaling_factor = tf.get_variable(name='Scaling_factor', shape=(subbnad_num, wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(scaling_factor)

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """

    def __init__(self, csm, mask, lam):
        with tf.name_scope('Ainit'):
            s = tf.shape(mask)
            self.nrow, self.ncol = s[0], s[1]
            print('nrow:', s[0], 'ncol:', s[1])
            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)
            self.lam = lam

    def myAtA(self, img):
        with tf.name_scope('AtA'):
            coilImages = self.csm * img
            kspace = tf_fftshift(tf.fft2d(tf_fftshift(coilImages))) / self.SF
            temp = kspace * self.mask
            coilImgs = tf_fftshift(tf.ifft2d(tf_fftshift(temp))) * self.SF
            coilComb = tf.reduce_sum(coilImgs * tf.conj(self.csm), axis=0)
            coilComb = coilComb + self.lam * img
        return coilComb


def myCG(A, rhs, x):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs = r2c_tf(rhs)
    x = r2c_tf(x)
    cond = lambda i, *_: tf.less(i, 5)

    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap))
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float(tf.reduce_sum(tf.conj(r) * r))
            beta = rTrNew / rTr
            beta = tf.complex(beta, 0.)
            p = r + beta * p
        return i + 1, rTrNew, x, r, p

    r_and_p = rhs - A.myAtA(x)
    i, r, p = 0, r_and_p, r_and_p
    rTr = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    return c2r_tf(out)


class Uclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """

    def __init__(self, csm, mask):
        with tf.name_scope('Uinit'):
            s = tf.shape(mask)
            self.nrow, self.ncol = s[0], s[1]
            print('shape of mask', tf.shape(mask))

            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)

    def myUlAtA(self, img):
        with tf.name_scope('ULAtA'):
            coilImages = self.csm * img
            kspace = tf_fftshift(tf.fft2d(tf_fftshift(coilImages))) / self.SF
        return kspace


def dc_unsupervised(rhs, csm, mask):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    rhs = r2c_tf(rhs)

    def fn(tmp):
        c, m, r = tmp
        Aobj = Uclass(c, m)
        y = Aobj.myUlAtA(r)
        return y

    inp = (csm, mask, rhs)
    rec = tf.map_fn(fn, inp, dtype=tf.complex64, name='valmapFn')
    return c2r_tf(rec)


def dc(rhs, csm, mask, lam1, x):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2 = tf.complex(lam1, 0.)

    def fn(tmp):
        c, m, r, x = tmp
        Aobj = Aclass(c, m, lam2)
        y = myCG(Aobj, r, x)
        return y

    inp = (csm, mask, rhs, x)
    rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
    return rec

def soft_threshold(z, T, U):
    """
       This function performs soft thresholding with parameter T for complex values
    """
    size_z = z.shape

    z = tf.reshape(z, [-1])
    threshold = tf.reshape(tf.math.multiply(T, U), [-1])

    # intermediate values
    a = tf.math.maximum(tf.math.abs(z) - threshold, 0)
    b = a + threshold
    c = tf.math.divide(a, b)
    d = tf.math.multiply(tf.cast(c, tf.complex64), z)

    sz = tf.reshape(d, size_z)

    return sz

def subband_procesing(u, temp_shape0, temp_shape1):
    prev_temp_shape0 = temp_shape0
    prev_temp_shape1 = temp_shape1
    temp_shape0 = temp_shape0 // 2
    temp_shape1 = temp_shape1 // 2

    u1 = u[temp_shape0: prev_temp_shape0, temp_shape1: prev_temp_shape1]
    u2 = u[0: temp_shape0, temp_shape1: prev_temp_shape1]
    u3 = u[temp_shape0: prev_temp_shape0, 0: temp_shape1]

    return u1, u2, u3, temp_shape0, temp_shape1

def subband_threshold(ui, U_list, umax, scaling_factor):
    T = tf.math.multiply(scaling_factor, umax**2)
    ui = soft_threshold(ui, T, U_list)

    return ui

def subband_reformation(temp, u3, u2, u1):
    temp1 = tf.concat([temp, u3], axis=0)
    temp2 = tf.concat([u2, u1], axis=0)

    return tf.concat([temp1, temp2], axis=1)

def calc_transform(x, hp, lp, wavelet_type, j):
    input_signal1 = tf.expand_dims(x[j, :, :, 0], -1)
    input_signal2 = tf.expand_dims(x[j, :, :, 1], -1)

    wave1 = nodes.dwt2d(input_signal1, lp, hp, wavelet=wavelet_type,
                        levels=level)
    wave2 = nodes.dwt2d(input_signal2, lp, hp, wavelet=wavelet_type,
                        levels=level)
    u = tf.squeeze(r2c_tf(tf.stack([wave1, wave2], axis=-1)))

    return u

def calc_reweighting(Wx):
    numerator = tf.ones_like(Wx, dtype=tf.float32)
    denominator = tf.math.abs(Wx) + tf.constant(eps, dtype=tf.float32)
    U = tf.math.divide(numerator, denominator)

    return U

def get_u_subbands(u):
    u_list = [0 for _ in range(subbnad_num)]
    temp_shape0 = u.shape[0]
    temp_shape1 = u.shape[1]

    for k in range(level):
        [u_list[3 * k + 0], u_list[3 * k + 1], u_list[3 * k + 2], temp_shape0, temp_shape1] = subband_procesing(u, temp_shape0, temp_shape1)
    u_list[12] = u[0: temp_shape0 // 2, 0: temp_shape1]
    u_list[13] = u[temp_shape0 // 2: temp_shape0, 0: temp_shape1]

    return u_list

def calc_threshold(x, hp, lp, wavelet_type):
    """
        This function gets inf norm of each subband of W_l*x
    """
    umax_list = [[0 for _ in range(subbnad_num)] for _ in range(batchSize)]
    for j in range(batchSize):
        u = calc_transform(x, hp, lp, wavelet_type, j)
        u_list = get_u_subbands(u)

        for k in range(subbnad_num):
            umax_list[j][k] = tf.math.reduce_max(tf.math.abs(u_list[k]))

    return umax_list

def Z_M_steps(umax, x, coeffs_decomp_lp, coeffs_decomp_hp, coeffs_recon_lp, coeffs_recon_hp, wavelet_type, beta, etta, v, i, l, U):
    """
       This function performs thresholding in wavelet domain (regularization) and dual updates
    """
    zl_returned = [0 for _ in range(batchSize)]
    for j in range(batchSize):
        beta_temp = beta[l][j]
        v_temp = v[l][j]

        u = calc_transform(x, coeffs_decomp_hp, coeffs_decomp_lp, wavelet_type, j)

        if i != 0:
            temp = c2r_tf(u - v_temp)
            beta_temp = beta_temp + tf.squeeze(r2c_tf(tf.stack([tf.math.multiply(etta[l], temp[...,0]), tf.math.multiply(etta[l], temp[...,1])], axis=-1)))
        u = u + beta_temp

        u_list = get_u_subbands(u)
        U_list = get_u_subbands(U[l][j])

        scaling_factor = getScaling_factor()

        for k in range(subbnad_num):
            u_list[k] = subband_threshold(u_list[k], U_list[k], umax[j][k], scaling_factor[k, l])

        v_temp = tf.concat([u_list[12], u_list[13]], axis=0)

        for k in range(level):
            v_temp = subband_reformation(v_temp, u_list[11-3*k], u_list[10-3*k], u_list[9-3*k])

        input_to_X = c2r_tf(v_temp - beta_temp)

        input_signal3 = tf.expand_dims(input_to_X[...,0], -1)
        input_signal4 = tf.expand_dims(input_to_X[...,1], -1)

        img3 = nodes.idwt2d(input_signal3, coeffs_recon_lp, coeffs_recon_hp, wavelet=wavelet_type,
                                 levels=level)
        img4 = nodes.idwt2d(input_signal4, coeffs_recon_lp, coeffs_recon_hp, wavelet=wavelet_type,
                                 levels=level)

        x_new = tf.concat([img3, img4], axis=-1)

        beta[l][j] = beta_temp
        v[l][j] = v_temp

        if j == 0:
            zl_returned = tf.expand_dims(x_new, 0)
        else:
            zl_returned = tf.concat([zl_returned, tf.expand_dims(x_new, 0)], axis=0)

    return zl_returned, beta, v

class UnrolledNet():
    def __init__(self, input_x, recon_x, sens_maps, mask, nb_blocks, training):
        self.input_x = input_x
        self.recon_x = recon_x
        self.sens_maps = sens_maps
        self.mask = mask
        self.nb_blocks = nb_blocks
        self.training = training

        self.model = self.Unrolled()

    def Unrolled(self):
        # wavelet coeffs
        d4_coeffs_decomp_hp = tf.constant([-0.010597401784997278,
                                           -0.032883011666982945,
                                           0.030841381835986965,
                                           0.18703481171888114,
                                           -0.02798376941698385,
                                           -0.6308807679295904,
                                           0.7148465705525415,
                                           -0.23037781330885523])

        d4_coeffs_decomp_lp = tf.constant([0.23037781330885523,
                                           0.7148465705525415,
                                           0.6308807679295904,
                                           -0.02798376941698385,
                                           -0.18703481171888114,
                                           0.030841381835986965,
                                           0.032883011666982945,
                                           -0.010597401784997278])

        d4_coeffs_recon_hp = tf.constant([-0.23037781330885523,
                                          0.7148465705525415,
                                          -0.6308807679295904,
                                          -0.02798376941698385,
                                          0.18703481171888114,
                                          0.030841381835986965,
                                          -0.032883011666982945,
                                          -0.010597401784997278])

        d4_coeffs_recon_lp = tf.constant([-0.010597401784997278,
                                          0.032883011666982945,
                                          0.030841381835986965,
                                          -0.18703481171888114,
                                          -0.02798376941698385,
                                          0.6308807679295904,
                                          0.7148465705525415,
                                          0.23037781330885523])

        d3_coeffs_decomp_hp = tf.constant([0.035226291882100656,
                                           0.08544127388224149,
                                           -0.13501102001039084,
                                           -0.4598775021193313,
                                           0.8068915093133388,
                                           -0.3326705529509569])

        d3_coeffs_decomp_lp = tf.constant([0.3326705529509569,
                                           0.8068915093133388,
                                           0.4598775021193313,
                                           -0.13501102001039084,
                                           -0.08544127388224149,
                                           0.035226291882100656])

        d3_coeffs_recon_hp = tf.constant([-0.3326705529509569,
                                          0.8068915093133388,
                                          -0.4598775021193313,
                                          -0.13501102001039084,
                                          0.08544127388224149,
                                          0.035226291882100656])

        d3_coeffs_recon_lp = tf.constant([0.035226291882100656,
                                          -0.08544127388224149,
                                          -0.13501102001039084,
                                          0.4598775021193313,
                                          0.8068915093133388,
                                          0.3326705529509569])

        d2_coeffs_decomp_hp = tf.constant([-0.12940952255092145,
                                           -0.22414386804185735,
                                           0.836516303737469,
                                           -0.48296291314469025])

        d2_coeffs_decomp_lp = tf.constant([0.48296291314469025,
                                           0.836516303737469,
                                           0.22414386804185735,
                                           -0.12940952255092145])

        d2_coeffs_recon_hp = tf.constant([-0.48296291314469025,
                                          0.836516303737469,
                                          -0.22414386804185735,
                                          -0.12940952255092145])

        d2_coeffs_recon_lp = tf.constant([-0.12940952255092145,
                                          0.22414386804185735,
                                          0.836516303737469,
                                          0.48296291314469025])

        haar_coeffs_decomp_hp = tf.constant([0.70710677, -0.70710677])

        haar_coeffs_decomp_lp = tf.constant([0.70710677, 0.70710677])

        haar_coeffs_recon_hp = tf.constant([-0.70710677, 0.70710677])

        haar_coeffs_recon_lp = tf.constant([0.70710677, 0.70710677])

        d4_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d4_coeffs_decomp_hp, -1), -1)
        d4_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d4_coeffs_decomp_lp, -1), -1)
        d4_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d4_coeffs_recon_hp, -1), -1)
        d4_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d4_coeffs_recon_lp, -1), -1)

        d3_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d3_coeffs_decomp_hp, -1), -1)
        d3_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d3_coeffs_decomp_lp, -1), -1)
        d3_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d3_coeffs_recon_hp, -1), -1)
        d3_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d3_coeffs_recon_lp, -1), -1)

        d2_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d2_coeffs_decomp_hp, -1), -1)
        d2_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d2_coeffs_decomp_lp, -1), -1)
        d2_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d2_coeffs_recon_hp, -1), -1)
        d2_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d2_coeffs_recon_lp, -1), -1)

        haar_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(haar_coeffs_decomp_hp, -1), -1)
        haar_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(haar_coeffs_decomp_lp, -1), -1)
        haar_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(haar_coeffs_recon_hp, -1), -1)
        haar_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(haar_coeffs_recon_lp, -1), -1)

        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(self.nb_blocks)]
        lam_init = tf.constant(0., dtype=tf.float32)
        x0 = dc(self.input_x,self.sens_maps ,self.mask, lam_init, x)

        beta = [[tf.zeros((nrow_GLOB, ncol_GLOB), dtype=tf.complex64) for _ in range(batchSize)] for _ in range(wavenum)]
        v = [[tf.zeros((nrow_GLOB, ncol_GLOB), dtype=tf.complex64) for _ in range(batchSize)] for _ in range(wavenum)]
        Wx = [[tf.zeros((nrow_GLOB, ncol_GLOB), dtype=tf.complex64) for _ in range(batchSize)] for _ in range(wavenum)]
        U = [[tf.zeros((nrow_GLOB, ncol_GLOB), dtype=tf.complex64) for _ in range(batchSize)] for _ in range(wavenum)]

        for j in range(batchSize):
            Wx[0][j] = calc_transform(self.recon_x, d4_coeffs_decomp_hp, d4_coeffs_decomp_lp, wavelet_type1, j)
            Wx[1][j] = calc_transform(self.recon_x, d3_coeffs_decomp_hp, d3_coeffs_decomp_lp, wavelet_type2, j)
            Wx[2][j] = calc_transform(self.recon_x, d2_coeffs_decomp_hp, d2_coeffs_decomp_lp, wavelet_type3, j)
            Wx[3][j] = calc_transform(self.recon_x, haar_coeffs_decomp_hp, haar_coeffs_decomp_lp, wavelet_type4, j)
            for l in range(wavenum):
                U[l][j] = calc_reweighting(Wx[l][j])

        umaxd4 = calc_threshold(x0, d4_coeffs_decomp_hp, d4_coeffs_decomp_lp, wavelet_type1)
        umaxd3 = calc_threshold(x0, d3_coeffs_decomp_hp, d3_coeffs_decomp_lp, wavelet_type2)
        umaxd2 = calc_threshold(x0, d2_coeffs_decomp_hp, d2_coeffs_decomp_lp, wavelet_type3)
        umaxd1 = calc_threshold(x0, haar_coeffs_decomp_hp, haar_coeffs_decomp_lp, wavelet_type4)

        with tf.name_scope('myModel'):
            with tf.variable_scope('Wts', reuse=tf.AUTO_REUSE):
                for i in range(self.nb_blocks):
                    # regularization and dual update
                    if i == 0:
                        etta = tf.constant(0., shape=(wavenum, 1), dtype=tf.float32)
                    else:
                        etta = getEtta()

                    # z1
                    [z1, beta, v] = Z_M_steps(umaxd4, x, d4_coeffs_decomp_lp,
                                                         d4_coeffs_decomp_hp, d4_coeffs_recon_lp, d4_coeffs_recon_hp,
                                                         wavelet_type1, beta, etta, v, i, 0, U)

                    # z2
                    [z2, beta, v] = Z_M_steps(umaxd3, x, d3_coeffs_decomp_lp,
                                                         d3_coeffs_decomp_hp,
                                                         d3_coeffs_recon_lp, d3_coeffs_recon_hp,
                                                         wavelet_type2, beta, etta, v, i, 1, U)

                    # z3
                    [z3, beta, v] = Z_M_steps(umaxd2, x, d2_coeffs_decomp_lp,
                                                         d2_coeffs_decomp_hp,
                                                         d2_coeffs_recon_lp, d2_coeffs_recon_hp,
                                                         wavelet_type3, beta, etta, v, i, 2, U)

                    # z4
                    [z4, beta, v] = Z_M_steps(umaxd1, x, haar_coeffs_decomp_lp,
                                                         haar_coeffs_decomp_hp,
                                                         haar_coeffs_recon_lp, haar_coeffs_recon_hp,
                                                         wavelet_type4, beta, etta, v, i, 3, U)

                    lam = getLambda()

                    denoiser_output = lam[0] * z1 + lam[1] * z2 + lam[2] * z3 + lam[3] * z4

                    rhs = self.input_x + denoiser_output

                    # dc
                    x = dc(rhs,self.sens_maps ,self.mask, lam[0] + lam[1] + lam[2] + lam[3], dc_output)
                    dc_output = x
                    # ...................................................................................................
                    all_intermediate_results[i][0] = r2c_tf(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = r2c_tf(tf.squeeze(dc_output))

                ul_output = dc_unsupervised(x, self.sens_maps, self.mask)
                scaling_factor = getScaling_factor()
            return x, ul_output, x0, all_intermediate_results, lam, scaling_factor, etta


def sense1(input_kspace, sens_maps):
    [m, n, nc] = np.shape(sens_maps)
    image_space = ifft(input_kspace, axes=(0, 1), norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    Eh_op = np.sum(Eh_op, axis=2)

    return Eh_op


# tf.reset_default_graph()
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# --------------------------------------------------------------------------
# %%Generate a meaningful filename to save the trainined models for testing
print('*************************************************')
start_time = time.time()
saveDir = 'savedModels/'
cwd = os.getcwd()  # returns current working directory of a process
directory = saveDir + \
            data_tag + str(acc_rate) + 'R_' + str(nb_blocks) + 'K_' + 'RB_' + str(
    epochs) + 'E_' + str(batchSize) + 'BS_' + LR + 'KspaceLoss'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'

# %% save test model
tf.reset_default_graph()
csmT = tf.placeholder(tf.complex64, shape=(None, ncoil_GLOB, nrow_GLOB, ncol_GLOB), name='csm')
maskT = tf.placeholder(tf.complex64, shape=(None, nrow_GLOB, ncol_GLOB), name='mask')
atbT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB, ncol_GLOB, 2), name='atb')
reconT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB, ncol_GLOB, 2), name='recon')

nw_out, ul_output, x0, all_intermediate_outputs, lam, scaling_factor, etta = UnrolledNet(atbT, reconT, csmT, maskT, nb_blocks,
                                                                   False).model  # False for is_training(BN)
out = tf.identity(nw_out, name='out')
ul_output = tf.identity(ul_output, name='predTst')
all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs')
x0 = tf.identity(x0, name='x0')
lam = tf.identity(lam, name='lam')
scaling_factor = tf.identity(scaling_factor, name='scaling_factor')
etta = tf.identity(etta, name='etta')
sessFileNameTst = directory + '/modelTst'

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
print('testing model saved:' + savedFile)

# %% read multi-channel dataset

# .......................Load the Data...........................................
print('Loading training data...')
kspace_train = h5.File(data_dir, "r")['kspace'][:] #change the internal variable name if necessary
trnCsm = h5.File(coil_dir, "r")['trnCsm'][:] #change the internal variable name if necessary
rand_masking = sio.loadmat(mask_dir)['mask'] #change the internal variable name if necessary
trnRecon = sio.loadmat(reweight_dir)['recon']

print('size of kspace: ', np.shape(kspace_train), ', maps: ', np.shape(trnCsm), ', mask: ', rand_masking.shape)

nSlice, nrow, ncol, ncoil = kspace_train.shape
trnMask = np.empty((nSlice, nrow, ncol), dtype=np.complex64)
trnMask = np.tile(rand_masking[np.newaxis, :, :], (nSlice, 1, 1))
"""
Pad the edges of the mask
"""
if data_opt == 'Coronal-PD' or data_opt == 'Coronal-PDFS':
    trnMask[:, :, 0:18] = np.ones((nSlice, nrow, 18))
    trnMask[:, :, 350:368] = np.ones((nSlice, nrow, 18))

temp = np.squeeze(kspace_train[:, :, :, 0])
trnOrg = np.empty(temp.shape, dtype=np.complex64)
trnAtb = np.empty(temp.shape, dtype=np.complex64)
print('getting the refs and aliased sense1 images')
# normalize k-space such that Atb is 0-255 region
for ii in range(np.shape(trnOrg)[0]):
    proc_mask = trnMask[ii]
    proc_mask = np.tile(proc_mask[:, :, np.newaxis], (1, 1, ncoil))
    sub_kspace = kspace_train[ii] * proc_mask
    sense_recon = lambda z: sense1(z, trnCsm[ii, ...])
    temp_zero_filled = sense_recon(sub_kspace)
    maximum = np.max(np.abs(temp_zero_filled[:]))
    factor = 255/maximum
    kspace_train[ii, ...] = kspace_train[ii, ...]*factor

    trnOrg[ii] = sense_recon(kspace_train[ii, ...])
    sub_kspace = kspace_train[ii] * proc_mask
    trnAtb[ii] = sense_recon(sub_kspace)

trnOrg, trnAtb, trnRecon = c2r(trnOrg), c2r(trnAtb), c2r(trnRecon)
trnCsm = np.transpose(trnCsm, (0, 3, 1, 2))
kspace_train = np.transpose(kspace_train, (0, 3, 1, 2))
kspace_train_ref = c2r(kspace_train)
print('size of the sensitivities', np.shape(trnCsm))
# creating the dataset
nTrn = trnAtb.shape[0]
total_batch = int(np.floor(np.float32(nTrn) / (batchSize * len(num_gpus))))

# Place all ops on CPU by default
tf.reset_default_graph()
with tf.device('/cpu:0'):
    # tf.reset_default_graph()
    tower_grads = []
    kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='ref_kspace')
    reconP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB, ncol_GLOB, 2), name='recon')
    csmP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='csm')
    maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='mask')
    atbP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB, ncol_GLOB, 2), name='atb')
    orgP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB, ncol_GLOB, 2), name='org')

    # creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices((kspaceP, reconP, orgP, atbP, csmP, maskP))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=trnOrg.shape[0])
    dataset = dataset.batch(batchSize)
    dataset = dataset.prefetch(len(num_gpus))
    iterator = dataset.make_initializable_iterator()
    for i in range(len(num_gpus)):
        with tf.device(assign_to_device('/gpu:{}'.format(num_gpus[i]), ps_device='/cpu:0')):
            kspaceT, reconT, orgT, atbT, csmT, maskT = iterator.get_next('getNext')
            # %% make training model
            out, ul_output, cg_out, all_intermediate_results, lam_out, scaling_factor_out, etta_out = UnrolledNet(atbT, reconT, csmT, maskT, nb_blocks,
                                                                                    True).model  # True indicates is_training
            out = tf.identity(out, name='nw_out')
            predT = tf.identity(ul_output, name='pred')
            scalar = tf.constant(0.5, dtype=tf.float32)
            # loss = tf.multiply(scalar,tf.norm(orgT-out)/tf.norm(orgT)) + tf.multiply(scalar,tf.norm(orgT-out,ord=1)/tf.norm(orgT,ord=1))
            loss = tf.multiply(scalar, tf.norm(kspaceT - ul_output) / tf.norm(kspaceT)) + tf.multiply(scalar, tf.norm(
                kspaceT - ul_output, ord=1) / tf.norm(kspaceT, ord=1))
            all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss)
            tower_grads.append(grads)
    # training code
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)
    print('parameters are: Epochs:', epochs, ' BS:', batchSize, 'nblocks:', nb_blocks)
    saver = tf.train.Saver(max_to_keep=50)
    totalLoss, totalTime, ep = [], [], 0
    avg_cost = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        feedDict = {kspaceP: kspace_train_ref, reconP: trnRecon, orgP: trnOrg, atbP: trnAtb, maskP: trnMask, csmP: trnCsm}
        sess.run(iterator.initializer, feed_dict=feedDict)
        savedFile = saver.save(sess, sessFileName)
        print("Model meta graph saved in::%s" % savedFile)
        print('Number of Parameters')
        print(sess.run(all_trainable_vars))
        print('Training...')
        for ii in range(1, epochs + 1):
            avg_cost = 0
            tic = time.time()
            for jj in range(total_batch):
                tmp, _, _, output = sess.run([loss, update_ops, train_op, out])
                avg_cost += tmp / total_batch
                if (ii == 1 and jj == 0):
                    print('Iter: ', ii, 'Loss : ', tmp)
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            totalTime.append(toc)
            print("Epoch:", ii, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))
            if (np.mod(ii, 10) == 0 or ii == 75 or ii == 85 or ii == 95 or ii == 98):
                saver.save(sess, sessFileName, global_step=ii)
                sio.savemat((directory + '/TrainingLog.mat'), {'loss': totalLoss})  # save the results
        saver.save(sess, sessFileName, global_step=ii)
        end_time = time.time()
        print('Training completed in minutes ', ((end_time - start_time) / 60))
        print('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
        print('*************************************************')
