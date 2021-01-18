# Style Transfer GAN Demo Application

### Overview
Information on the fast artistic style transfer can be found in the [article](https://cs.stanford.edu/people/jcjohns/eccv16/).

The main idea behind the project is to make an image stylization service (analogue of Prisma app).

The mathematical problem statement: Fast Neural Style generative adversarial network (GAN) with L2-loss between the generated and the original images to preserve the original image content and with L2-loss between the images' Gram matrices to preserve the given style.

### Usage:
The app can be tested by running the following commands:
```
git clone https://github.com/Illumaria/style-transfer-GAN
cd style-transfer-GAN/
pip install -r requirements.txt
FLASK_APP=main.py flask run
```
After that, follow the link provided in the terminal.
