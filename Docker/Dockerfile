FROM ubuntu:16.04
MAINTAINER yx59@duke.edu

ENV PACKAGES git make gcc g++ gfortran locales \
        libhdf5-dev libhdf5-serial-dev \
        libboost-all-dev \
        gsl-bin libgsl-dbg libgsl-dev \
        python3-dev liblapack-dev libatlas-base-dev \
        python3-numpy python3-scipy python3-h5py cython3 \
        python3-setuptools python3-pip

RUN apt-get update &&  \
    apt-get -y install build-essential wget

# install cmake-3.4.0
RUN mkdir /usr/local/cmake
WORKDIR /usr/local/cmake
RUN wget http://www.cmake.org/files/v3.4/cmake-3.4.0.tar.gz  
RUN tar xf cmake-3.4.0.tar.gz
WORKDIR /usr/local/cmake/cmake-3.4.0
RUN ./configure && make
ENV PATH="/usr/local/cmake/cmake-3.4.0/bin:${PATH}"

# install libhdf5, boos, gsl, python3 related
RUN apt-get install -y --no-install-recommends ${PACKAGES}
RUN pip3 install --upgrade scipy

# set the locale language to UTF-8 encoding
RUN locale-gen en_US.UTF-8
ENV LANG="en_US.UTF-8"

# download and install hic_HQ-osg
RUN mkdir /usr/local/scratch 
WORKDIR /usr/local/scratch
RUN git clone --recursive https://github.com/Yingru/hic_HQ-osg.git

WORKDIR /usr/local/scratch/hic_HQ-osg/
RUN git checkout freestreaming && \
        git submodule init && \
        git submodule update

RUN bash makepkg
RUN cp hic_HQ-osg.tar.gz /usr/local

# run the events
WORKDIR /usr/local
RUN tar -xzf hic_HQ-osg.tar.gz

ENV PYTHONPATH="/usr/local/hic_HQ-osg/lib/python3.5/site-packages:/usr/local/hic_HQ-osg/local/lib/python3.5/dist-packages"
ENV PATH="/usr/local/hic_HQ-osg/bin:$PATH"
ENV XDG_DATA_HOME="/usr/local/hic_HQ-osg/share"

#WORKDIR /usr/local/hic_HQ-osg/results
