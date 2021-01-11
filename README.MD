# Architecture Search

cd sota/cnn  && python train_search.py

# Architecture Train

Train CNN cells on cifar10 and cifar 100: cd sota/cnn && python train.py --auxiliary --cutout

# Architecture Evaluation

Test on cifar10: cd sota/cnn && python test_c10.py --auxiliary
Test on cifar100: cd sota/cnn && python test_c100.py --auxiliary

