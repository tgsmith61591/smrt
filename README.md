[![Build status](https://travis-ci.org/tgsmith61591/smrt.svg?branch=master)](https://travis-ci.org/tgsmith61591/smrt)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg) 
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)


# Synthetic Minority Reconstruction Technique (SMRT)
*Handle your class imbalance more intelligently by using SMOTE's younger, more sophisticated cousin*


### Installation

Installation is easy. After cloning the project onto your machine and installing the required dependencies,
simply use the `setup.py` file:

```bash
$ git clone https://github.com/tgsmith61591/smrt.git
$ cd smrt
$ python setup.py install
```

### About

SMRT (Sythetic Minority Reconstruction Technique) is the new SMOTE (Synthetic Minority Oversampling TEchnique).
Using variational auto-encoders, SMRT learns the latent factors that best reconstruct the observations in each
minority class, and then generates synthetic observations until the minority class is represented at a user-defined
ratio in relation to the majority class size.

SMRT avoids one of SMOTE's greatest risks: In SMOTE, when drawing random observations from whose k-nearest
neighbors to synthetically reconstruct, the possibility exists that a "border point," or an observation very close to
the decision boundary may be selected. This could result in the synthetically-generated observations lying
too close to the decision boundary for reliable classification, and could lead to the degraded performance
of an estimator. SMRT avoids this risk implicitly, as the [``VariationalAutoencoder``](smrt/autoencode/autoencoder.py)
learns a distribution that is generalizable to the lowest-error (i.e., most archetypal) observations.

### Notes

- See [examples](examples/) for usage
- Information on [the authors](AUTHORS.md)
- S-M-R-T [original reference](https://www.youtube.com/watch?v=tcGQpjCztgA) (enable audio)
