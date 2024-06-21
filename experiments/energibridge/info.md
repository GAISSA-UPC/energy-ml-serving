##
verify
ls -la /dev/cpu/*/msr

## install windows
- install rust, ( install the Visual Studio C++ Build tools: https://www.rust-lang.org/tools/install)

./energibridge[.exe] --max-execution 10 --output energy.csv --command-output Chrome.log --interval 200 -- google-chrome google.com

./energibridge[.exe] --max-execution 10 --output energy.csv --command-output Chrome.log --interval 200 echo "hello_y"

## linux
from: https://github.com/tdurieux/EnergiBridge

git clone https://github.com/tdurieux/EnergiBridge

sudo chgrp -R msr /dev/cpu/*/msr;
sudo chmod g+r /dev/cpu/*/msr;

cargo build -r;

sudo setcap cap_sys_rawio=ep target/release/energibridge;