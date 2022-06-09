import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#nvidia-smi
import numpy as np
import tensorflow as tf
import scipy.io as sio
from skimage.measure import compare_ssim
import h5py as h5
import time

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
acc_rate=4
data_opt = 'Coronal-PD'

# user should specify the directories for data, coil, mask, file directory, save directory, and reweight_dir
data_dir = "..."
coil_dir = "..."
mask_dir = '...'
file_dir = "..." #the directory for the folder containing trained model
save_dir = '....mat' #save the output reconstructions as mat file in this directory
reweight_dir = '....mat' #the directory of the first set of reconstructed images
nrow_GLOB, ncol_GLOB, ncoil_GLOB = 320, 368, 15 #change the dimensions of the dataset if necessary

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

def sense1(input_kspace,sens_maps):
    [m,n,nc]= np.shape(sens_maps)
    image_space =ifft(input_kspace,axes=(0, 1), norm=None, unitary_opt=True)
    Eh_op =np.conj(sens_maps)*image_space
    Eh_op = np.sum(Eh_op,axis=2)

    return Eh_op

def myNMSE(ref,recon):
    """ This function calculates NMSE between the original and
    the reconstructed images"""
    ref,recon=np.abs(ref),np.abs(recon)
    nrmse=np.linalg.norm(ref-recon)/np.linalg.norm(ref)
    nmse = nrmse**2
    return nmse

def mySSIM(space_ref,space_rec):
    space_ref = np.squeeze(space_ref)
    space_rec = np.squeeze(space_rec)
    space_ref = space_ref / np.amax(np.abs(space_ref))
    space_rec = space_rec / np.amax(np.abs(space_ref))
    data_range = np.amax(np.abs(space_ref)) - np.amin(np.abs(space_ref))
    return compare_ssim(space_rec, space_ref, data_range=data_range,
                                              gaussian_weights=True,
                                              use_sample_covariance=False)

def myPSN(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed images"""
    print('org size', org.size)
    print('org max',np.abs(org.max()))
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(np.abs(org.max())/(np.sqrt(mse)+1e-10 ))
    return psnr

def norm_rssq(tensor, axes=(0, 1, 2), keepdims=True):
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
    if not keepdims: return tensor.squeeze()

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

cwd=os.getcwd()
tf.reset_default_graph()
subDirectory=file_dir
# read multi-channel dataset
print('loading '+ data_opt+ ' dataset...')
kspace_test=h5.File(data_dir, "r")['kspace'][:] #change the internal variable name if necessary
testCsm = h5.File(coil_dir, "r")['testCsm'][:] #change the internal variable name if necessary
print('size of the training data: ',np.shape(kspace_test),', size of the sensitivities: ',np.shape(testCsm))
print('testing')
nSlice,nrow,ncol,ncoil=kspace_test.shape
sensitivities =np.copy(testCsm)

print('getting the validation and training masks...')
origMask=np.empty((nSlice,nrow,ncol),dtype=np.complex64)
random_masking=sio.loadmat(mask_dir)['mask'] #change the internal variable name if necessary
test_recon = sio.loadmat(reweight_dir)['recon']
origMask =np.tile(random_masking[np.newaxis,:,:],(nSlice,1,1))

if data_opt == 'Coronal-PD' or data_opt == 'Coronal-PDFS':
    origMask[:,:,0:18] =np.ones((nSlice,nrow,18))
    origMask[:,:,350:368] =np.ones((nSlice,nrow,18))

print('shape of Mask  : ',np.shape(origMask))

temp=np.squeeze(kspace_test[:,:,:,0])
testOrg=np.empty(temp.shape,dtype=np.complex64)
testAtb=np.empty(temp.shape,dtype=np.complex64)
ref_kspace=np.empty(kspace_test.shape,dtype=np.complex64)
print('ksp rescale')
for ii in range(np.shape(testOrg)[0]):
    proc_mask=origMask[ii]
    proc_mask=np.tile(proc_mask[:,:,np.newaxis],(1,1,ncoil))
    sub_kspace=kspace_test[ii]*proc_mask
    sense_recon= lambda z: sense1(z,testCsm[ii,...])
    temp_zero_filled = sense_recon(sub_kspace)
    maximum = np.max(np.abs(temp_zero_filled[:]))
    factor = 255/maximum
    kspace_test[ii, ...] = kspace_test[ii, ...]*factor

    testOrg[ii] = sense_recon(kspace_test[ii, ...])
    sub_kspace = kspace_test[ii] * proc_mask
    testAtb[ii] = sense_recon(sub_kspace)

testCsm=np.transpose(testCsm,(0,3,1,2))
testOrgAll=np.copy(testOrg)
testReconAll=np.copy(test_recon)
testAtbAll=np.copy(testAtb)
testCsmAll=np.copy(testCsm)
all_recon_slices=list()
all_recon_slices_postproc=list()
all_ref_slices=list()
all_input_slices=list()
all_ref_kspace = list()
all_input_kspace =list()

all_slices_intermediate_outputs=[]
dc_outputs =[]
print ('Now loading the model ...')
load_model_dir=subDirectory #complete path
tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(load_model_dir)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(load_model_dir+'/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    #..................................................................................................................
    graph = tf.get_default_graph()
    predT =graph.get_tensor_by_name('out:0')
    ul_output =graph.get_tensor_by_name('predTst:0')
    reconT = graph.get_tensor_by_name('recon:0')

    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs=graph.get_tensor_by_name('all_intermediate_outputs:0')

    #...................................................................................................................
    maskT =graph.get_tensor_by_name('mask:0')
    atbT=graph.get_tensor_by_name('atb:0')
    csmT=graph.get_tensor_by_name('csm:0')
    wts=sess.run(tf.global_variables())

    for ii in range(np.shape(testOrg)[0]):

        testOrg = np.copy(testOrgAll[ii, :, :])[np.newaxis]
        testAtb = np.copy(testAtbAll[ii, :, :])[np.newaxis]
        testRecon = np.copy(testReconAll[ii, :, :])[np.newaxis]
        testCsm = np.copy(testCsmAll[ii, :, :, :])[np.newaxis]
        testMask = np.copy(origMask[ii, :, :])[np.newaxis]
        testOrg, testAtb, testRecon = c2r(testOrg), c2r(testAtb), c2r(testRecon)

        dataDict = {atbT: testAtb, reconT: testRecon, maskT: testMask, csmT: testCsm}
        tic = time.time()
        rec, ul_out, x0, all_intermediate_outputs_temp=sess.run([predT, ul_output, x0_output,all_intermediate_outputs],feed_dict=dataDict)

        rec = r2c(rec.squeeze())

        rec = np.sqrt(np.sum(np.abs(testCsm[0, :, :, :]) ** 2, 0)) * rec

        rec = np.expand_dims(c2r(rec), 0)


        dataDict = {atbT: testAtb, reconT: rec, maskT: testMask, csmT: testCsm}
        rec, ul_out, x0, all_intermediate_outputs_temp = sess.run([predT, ul_output, x0_output, all_intermediate_outputs], feed_dict=dataDict)

        testOrg=r2c(testOrg.squeeze())
        testAtb=r2c(testAtb.squeeze())
        rec=r2c(rec.squeeze())
        x0=r2c(x0.squeeze())

        rec = np.sqrt(np.sum(np.abs(testCsm[0, :, :, :]) ** 2, 0)) * rec
        toc = time.time() - tic

        #...............................................................................................................
        print('elapsed time %f seconds' % toc)

        all_recon_slices.append(rec)
        all_ref_slices.append(testOrg)
        all_input_slices.append(testAtb)
        print('ITERATION --------------->',ii)

sio.savemat((save_dir),{'recon':all_recon_slices})