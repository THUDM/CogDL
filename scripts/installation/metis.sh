file="metis-5.1.0"

wget "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${file}.tar.gz"
gunzip "${file}.tar.gz"
tar -xvf "${file}.tar"

cd ${file}
make config shared=1 prefix="~/.local/"
make install
export METIS_DLL="~/.local/lib/libmetis.so"
echo $METIS_DLL
echo $PWD
ls ~/.local/lib
cd ..
