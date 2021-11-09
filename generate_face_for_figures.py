#%%
import copy

import numpy as np
from PIL import Image
from matplotlib import cm

from GFG import Ctx, GL
from GFG.model import Nf, Adata
from ogbGL.model import Material, Newmtl

#%%
adata = Adata.load('quick_FACS_blendshapes_v2dense')
nf = Nf.from_default()
nfmat = copy.deepcopy(nf.material)

#%%
ctx = Ctx()
nf.attach(ctx)
adata.attach(ctx)

#%%
# Darwin, fear
#AUs = ('AU1', 'AU2', 'AU5', 'AU20')
#cmaps = (cm.Blues, cm.Blues, cm.Reds, cm.Reds)
AUs = ('AU9',)
cmaps = (cm.Reds,)# cm.Reds, cm.Blues)
#amps = (1, 0.5)
#cmaps = (cm.Reds,)
amps = (1,)
threshold = 0.5

data = np.zeros((2048, 2048, 3))
denom = np.zeros((2048, 2048))
alpha = np.zeros((2048, 2048))

for i, (au, cmap, amp) in enumerate(zip(AUs, cmaps, amps)):
    # Set current AU to amplitude 1
    
    adata.bshapes[au] = amp

    # Draw AU movement
    # here we get the full precision of the dSim buffer
    ctx._blendshapes_pass()
    ctx._image_pass(adata.bshapes.fbo, 'dSim')
    im = np.dstack([np.asarray(im_.resize((2048, 2048), resample=Image.BICUBIC))
                    for im_ in ctx.im])[::-1]
    
    # Transform to mm
    im -= 0.5
    im *= adata.bshapes.dSim_mx

    # Create norm of deviation (across x, y, z) + 
    # normalize to 0-1 range
    dev = np.linalg.norm(im[:, :, :3], axis=2)
    dev -= dev.min() 
    dev /= dev.max()

    dev[dev < threshold] = 0

    # From 2048x2048 dev map to 2048x2048x4 RBGA colormap
    # with alpha proportional to dev
    dev_rgba = cmap(dev, alpha=dev)
    
    # denom stores a1+a2+...+an and is divided out after the loop
    a = dev_rgba[:, :, 3]
    data += dev_rgba[:, :, :3] * a[:, :, np.newaxis]
    alpha = np.maximum(alpha, a)
    denom += a

    adata.bshapes[au] = 0.

#%%
# divide out denom returning zeros in the output for zeros 
# in the denominator
with np.errstate(divide='ignore', invalid='ignore'):
    data = np.where(denom[:,:,np.newaxis] > 0.,
                    data / denom[:,:,np.newaxis],
                    0.)
# add full alpha channel and scale to 255
data = np.dstack((data, np.ones(data.shape[0:2]))) * 255.

#%%
# image data should be numpy array, float, (y,x,RGBA), 0->255
# alpha should be numpy array (y,x), 0->1
nmtdict = {'name': None,
           'maps': {'map_Kd': {'name': None,
                              'data': data,
                              'mask': alpha}}}

mymaterial = Material(newmtl = [Newmtl.from_dict(nmtdict)])
mymaterial.split_newmtl(mat_groups = nf.material_groups,
                        fvt = nf.face_tuple.fv,
                        vt = nf.vertex_tuple.dSvt,
                        groupfidx = nf.groupfindex)

nf.material.merge(mymaterial)

# now you need to reatttach nf to see the changes
nf.detach()
nf.attach(ctx)

#now we animate face
for au in AUs:
    adata.bshapes[au] = 1

ctx.render()

#%%
im = ctx.render(dest='image')
im.save('oltest.png')

