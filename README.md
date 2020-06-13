# GAN-Based Facial Attractiveness Enhancement 

This is the code repository for our manuscript: [https://arxiv.org/abs/2006.02766](https://arxiv.org/abs/2006.02766)

## README, cleared code and models will be coming soon.

#### InterFaceGAN ans StyleGAN Based Beautification
We use the [stylegan-ffhq-1024x1024.pkl](https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ) provided by StyleGAN repo.

Here is the beauty boundary we trained. [beauty boundary](https://drive.google.com/drive/folders/1VD1aKG9SgQ8GhyISdsScLB9QSKYzn4Nb?usp=sharing)

For the editing part, please refer to the InterFaceGAN repo. :) 

This repo contains modified Beholder-GAN, evaluation part and some should-be-cleared files.

#### Enhanced Version of Beholder-GAN

Here is our pretrained model for modified Beholder-GAN. [beholder-id.pkl](https://drive.google.com/file/d/1rUZ2bmXl0Re952l4QwO1s4cJjn3kKJ4C/view?usp=sharing)

For the model of Beholder-GAN, we trained the model using FFHQ instead of CelebA-HQ and due to limited training information Beholder-GAN paper provided, we changed mini batch size and set the resolution to 128x128 to fit our then machine with everything else untouched. Unfortunately, we cannot reproduce the same performance as Beholder-GAN paper.

## Datasets  
  
The datasets we worked on can be found in these links:  
* FFHQ: [FFHQ](https://github.com/NVlabs/ffhq-dataset)  
