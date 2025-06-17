# LINUX (UBUNTU)

1. Install RISC-V Toolchain
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install autoconf automake autotools-dev curl python3 \
    libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison \
    flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev

# Clone and build RISC-V GNU toolchain
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gc --with-abi=lp64d
make -j$(nproc)

# Add to PATH
echo 'export PATH=/opt/riscv/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```


2. Install Spike RISC-V Simulator

```bash
# Install Spike (the official RISC-V simulator)
git clone https://github.com/riscv/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/opt/riscv --with-fesvr=/opt/riscv
make -j$(nproc)
sudo make install
```

3. Get the MNIST Inference Code
```bash
git clone https://github.com/lftraining/chipyard-micro-learning.git
wget https://raw.githubusercontent.com/lftraining/chipyard-micro-learning.git/scripts/mnist-inference.c
```
