# RISC-V AI Chips
## A practical microcourse on building AI accelerators with Chipyard

### Installation Guide

### System Requirements

#### Minimum Hardware:

8GB RAM (16GB recommended)
20GB free disk space
x86_64 Linux system (Ubuntu 20.04+ recommended)

#### Supported Platforms:

Linux (Ubuntu, CentOS, Fedora)
WSL2 on Windows 10/11
macOS (with some limitations)

#### Step 1: Base System Setup
##### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y build-essential git curl wget
sudo apt install -y autoconf automake autotools-dev
sudo apt install -y libmpc-dev libmpfr-dev libgmp-dev
sudo apt install -y gawk bison flex texinfo gperf
sudo apt install -y libtool patchutils bc zlib1g-dev
sudo apt install -y device-tree-compiler pkg-config libglib2.0-dev
```
##### CentOS/RHEL/Fedora
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git curl wget
sudo yum install -y autoconf automake libtool
sudo yum install -y mpfr-devel gmp-devel libmpc-devel
sudo yum install -y gawk bison flex texinfo
sudo yum install -y zlib-devel device-tree-compiler
```

##### Step 2: Python Environment
```bash 
# Install Python 3.7+
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment for the course
python3 -m venv ~/riscv-course-env
source ~/riscv-course-env/bin/activate

# Install required Python packages
pip install numpy matplotlib jupyter
```
Step 3: RISC-V GNU Toolchain
```bash
# Create installation directory
sudo mkdir -p /opt/riscv
sudo chown $USER:$USER /opt/riscv

# Download pre-built toolchain (Ubuntu x86_64)
cd /opt/riscv
wget https://github.com/sifive/freedom-tools/releases/download/v2020.12.0/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14.tar.gz

# Extract
tar -xzf riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14.tar.gz

# Add to PATH
echo 'export PATH=/opt/riscv/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
##### Step 4: Spike RISC-V Simulator
```bash
# Clone and build Spike
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build
cd build
../configure --prefix=/opt/riscv
make -j$(nproc)
sudo make install

# Verify installation
spike --help
```
##### Step 5: Chipyard Framework
```bash
# Clone Chipyard (this will take several minutes)
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard

# Initialize submodules (this will take 15-30 minutes)
./build-setup.sh

# Build the default RISC-V toolchain for Chipyard
./scripts/build-toolchain.sh

# Source the Chipyard environment
source env.sh
```
##### Estimated Installation Time

- Fast internet + powerful machine: 1-2 hours
- Average setup: 2-3 hours
- Slower systems or internet: 3-4 hours

Most time is spent downloading and compiling Chipyard components.


## Profiling tools  [Installation guide](/guides/Profiling_guide])