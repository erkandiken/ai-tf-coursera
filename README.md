This repo includes rework of some exercises given in
Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning Coursera course.

Check the links below for the course and original github repo.

https://www.coursera.org/learn/introduction-tensorflow

https://github.com/lmoroney/dlaicourse


Create virtual environment, both folder and environment named as ENV,
replace your preferred name with ENV:

    mkdir <ENV>
    cd <ENV>
    virtualenv -p python3.6 .
    source /bin/activate

Install required packages, including TF 2.0 Alpha (https://www.tensorflow.org/alpha),
matplotlib, jupyter notebook:

    pip install -r requirements.txt

In order to able to run and create jupyter notebook with current
virtual environment, install kernelspec for current ENV:

    ipython kernel install --user --name=<ENV>


Now, you are ready to launch jupter notebook:

    jupyter notebook

After you are done, you can deactivate the virtual environment:

     deactivate

