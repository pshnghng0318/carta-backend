## Install xtl for xtensor
`brew install xtl`

## Clean build folders
```bash
rm -rf build
rm -rf third-party/tensorstore/build
```

## Install abseil
### Remove existing version
```bash
sudo rm -rf /usr/local/include/absl /usr/local/lib/libabsl*
```
 
### Install compatible version
```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
git fetch --all --tags
git checkout 20250814.1
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=17
make -j$(nproc)
sudo make install
sudo ldconfig
```