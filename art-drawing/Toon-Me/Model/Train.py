# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="V_1nUjYIhks1"
# # **Toon-Me**
#

# + id="8mOG3Ph8hji_"
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from torchvision.models import vgg16_bn
from fastai.utils.mem import *

# + [markdown] id="vI3M-s6Mh2Cz"
# **Path**
#

# + [markdown] id="YbJuATO9lFeq"
# **Line art and toon photos**

# + id="9QYvHJMGh4sp"
path = Path('/content/gdrive/My Drive/Colourizer')
path_hr = Path('/content/gdrive/My Drive/Colourizer/split colour')
path_lr= Path('/content/gdrive/My Drive/Colourizer/split draw')
path_hr1 = Path('/content/gdrive/My Drive/Colourizer/Colour')
path_lr1= Path('/content/gdrive/My Drive/Colourizer/Line')

# + [markdown] id="Imxh8Uz6iVGz"
# **Architecture**
#
#

# + [markdown] id="vTHNY-5ukeqf"
# **64px**

# + id="NIHhsDU5iBBC"
bs,size=20,64
arch = models.resnet34

# + [markdown] id="vhm3Gd11k8dY"
# **Portrait photos split by half**

# + id="ZPY5QKnaiCxD"
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.2, seed=42)
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs,num_workers = 0).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


# + id="3c3TuqPWiF13"
data = get_data(bs,size)
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))

# + id="f3U3Rz8siJT3"
t = data.valid_ds[0][1].data
t = torch.stack([t,t])


# + id="rHUOVM4KicVx"
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


# + id="QVgAEVcoieCM"
gram_matrix(t)


# + id="obfCa5cgif9C"
base_loss = F.l1_loss


# + id="g_mXAT8tii_H"
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

# + id="M76TNPrLiku6"
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


# + [markdown] id="YfBC3x2clbcW"
# **Feature Loss**

# + id="a1vExzUdilj3"
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


# + id="e_Lor9ioiojV"
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])


# + id="53lcHVZSiqNk"
wd = 1e-3
y_range = (-3.,3.)


# + id="BzeRODpZirwE"
def create_gen_learner():
    return unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();

# + id="XE2Em1YtiteL"
learn_gen = create_gen_learner()


# + id="A8Cx1_f0ixMT"
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="sgl9kt7bi2Of"
lr = 4.79E-04
epoch = 10
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn_gen.fit_one_cycle(epoch, lrs, pct_start=pct_start,)
    learn_gen.save(save_name)
    learn_gen.show_results(rows=1, imgsize=5)


# + id="fiSYxhhSi4JU"
do_fit('da', slice(lr*10))
#lr*10

# + id="JNhGBirhi56g"
learn_gen.unfreeze()
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="vOi1eomYi_pp"
epoch = 5
do_fit('db', slice(9.12E-07,lr))

# + [markdown] id="T4cpEx0qlQjT"
# **Line art and Toon portraits**

# + [markdown] id="cjQl5d7SkkDA"
# **64px**

# + id="U8X-fQq6jCsv"
src = ImageImageList.from_folder(path_lr1).split_by_rand_pct(0.2, seed=42)
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr1/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs,num_workers = 0).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


# + id="Rf1uhlhmjavZ"
data = get_data(20,64)
learn_gen.data = data
learn_gen.freeze()
gc.collect()
learn_gen.load('db');

# + id="6cuxYyIojhwE"
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))

# + id="xxyidcttjlzo"
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="rkm4LT9Xjnz1"
epoch = 5
do_fit('a2', slice(7.59E-05))

# + id="uSO-3DG3jqAO"
learn_gen.unfreeze()
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="OEwfbsWIjuzk"
epoch = 5
do_fit('a3', slice(9.12E-07,1e-5), pct_start=0.3)

# + [markdown] id="1ivzigFBkuNL"
# **128px**

# + id="-QF5i0-cjzvY"
data = get_data(8,128)
learn_gen.data = data
learn_gen.freeze()
gc.collect()
learn_gen.load('a3');

# + id="Sh82aN9jj1TV"
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="T9-DyqcUj3P7"
epoch = 5
do_fit('db2', slice(9.12E-07))

# + id="lMXIWNEBj5N1"
learn_gen.unfreeze()
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="gyJSQPzKkBvk"
epoch = 5
do_fit('db3', slice(6.31E-07,1e-5), pct_start=0.3)

# + [markdown] id="eMoO09I3kzZt"
# **192px**

# + id="YBQdCyFKkD98"
data = get_data(5,192)
learn_gen.data = data
learn_gen.freeze()
gc.collect()
learn_gen.load('db3');

# + id="baPQXLs9kGBG"
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="WNNpibe1kH9m"
epoch =5
lr = 1.32E-06
do_fit('db4')

# + id="Vfd12_-gkJUm"
learn_gen.unfreeze()
learn_gen.lr_find()
learn_gen.recorder.plot(suggestion =True)

# + id="6ltQR5QJkMQR"
epoch = 5
do_fit('db5', slice(6.31E-07,1e-5), pct_start=0.3)

# + id="oYq7OcMFkVTi"
learn_gen.show_results(rows=10)
