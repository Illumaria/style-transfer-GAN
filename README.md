# Style Transfer GAN Demo Application

### Overview
Information on the fast artistic style transfer can be found in the [article](https://cs.stanford.edu/people/jcjohns/eccv16/).

The main idea behind the project is to make an image stylization service (analogue of Prisma app).

The mathematical problem statement: Fast Neural Style generative adversarial network (GAN) with L2-loss between the generated and the original images to preserve the original image content and with L2-loss between the images' Gram matrices to preserve the given style.

### Usage:
The app can be tested either by the [link](https://machines-do-art.ew.r.appspot.com/) or by running the following command:
```
FLASK_APP=main.py flask run
```

