# this is the latest version that my gpu currently supports (min between linux and windows/wsl compat)
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# create a non-root user to use once podman gets fixed up
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
  && apt-get update \
  && apt-get install -y sudo \
  && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
  && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN apt update \
  && apt install -y \
    build-essential \
    curl \
    git

# RUN useradd -m -s /bin/bash linuxbrew && \
#   echo 'linuxbrew ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers


USER vscode
WORKDIR /home/vsocde

# # set up zsh because me like
# RUN apt update \
#   && apt install -y \
#     curl \
#     git
#     zsh \
#   && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# install homebrew
RUN NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
  && echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/vscode/.profile

RUN eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" \
  && brew install \
    tmux \
    vim \
    fftw \
    cmake \
    pkg-config


# # install helpful or neccessary packages (some of these may be replaced by homebrew once that's going)
# RUN apt update \
#     && apt install -y \
#         git \
#         tmux \
#         vim


# This breaks when using podman non-root (i think because it uses the host's UID/GID?) but I think is needed for windows?
# Comment out on podman!
# USER ${USERNAME}

# ENTRYPOINT ["/bin/bash", "-c"]