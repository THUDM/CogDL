file="metis-5.1.0"

wget https://cloud.tsinghua.edu.cn/f/fd30d629a08645bbbdb8/?dl=1 -O "{file}.tar.gz"
gunzip "${file}.tar.gz"
tar -xvf "${file}.tar"

cd ${file}
make config shared=1 prefix="~/.local/"
make install
