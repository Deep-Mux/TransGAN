# TransGAN

A quick guide on how to deploy TransGAN network on the cloud using DeepMux serverless functions.

In this example we use use weights published by the authors to generate CelebA-like images.

Full guide describing the deployment is availible at [Medium](https://deepmux.medium.com/how-to-deploy-transgan-for-generating-celeba-like-pictures-1537767e3295).

## Credits

Code uses original [TransGAN](https://github.com/VITA-Group/TransGAN) implementation referenced in [TransGAN: Two Transformers Can Make One Strong GAN", Yifan Jiang, Shiyu Chang, Zhangyang Wang](https://arxiv.org/abs/2102.07074) paper.

## Deploy and run

#### 1. Download model weights

Download `celeba64_checkpoint.pth` checkpoint from [Google Drive](https://drive.google.com/file/d/1M_wAaiIU4XYbge_GXNKeptH9WBeBGp3e/view?usp=sharing) to `./pretrained_weight` directory.

#### 2. Install and configure DeepMux cli
```shell script
pip install deepmux-cli

deepmux login
```

#### 3. Deploy the model
```shell script
./deploy.sh
```

#### 4. Run the model
```shell script
deepmux run --name TransGAN --data '' > face.jpg
```
OR
```shell script
curl -o face.jpg \
  -X POST \
  -H "X-Token: <YOUR API TOKEN>" \
  https://api.deepmux.com/v1/function/TransGAN/run
```
