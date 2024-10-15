FROM ubuntu:18.04 as builder
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>"

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
                        bash \
                        wget \
                        bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# create a working directory
RUN mkdir /app
WORKDIR /app

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-py37_4.9.2-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda install --yes cmake \
    && conda clean -ya

COPY app/requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache


# production image
FROM ubuntu:18.04

ENV TZ=America/Montreal

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY --from=builder /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

COPY app/ /app/

RUN chmod -R 777 /app && chmod +x /app/inference.py

CMD ["python3"]