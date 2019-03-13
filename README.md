# face-detection
Sample face detection using opencv

#### Langkah-langkah installasi sebelum coding
1. ```mkdir face-detection```
2. ```virtualenv --python=python3.7 face-detection```
3. ```source face-detection/bin/activate```
3. ```mkdir dataset```
4. Install all packet needed using command ```pip install -r requirements.txt```

#### file-file python yang bisa digunakan
1. `data-preparation.py` untuk membuat data model
2. `face-detect.py` untuk testing face detection menggunakan foto dan menulis hasil face detection ke dalam sebuah folder
3. `face-detect-live.py` untuk mengambil gambar face pada perangkat kamera langsung
4. `face-detect-single.py` untuk testing single gambar, optimize parameter pada face detection
5. `face-training.py` untuk training data, membuat model
6. `face-testing.py` untuk uji coba hasil training menggunakan live kamera

#### Jika konfigurasi dan instalasi selesai dilakukan, untuk dapat menjalankan langkah-langkahnya adalah sebagai berikut;
1. jalankan file `face-detect-live.py` sampai muncul tampilan windows untuk pengambilan gambar. aplikasi akan secara otomatis mengambil gambar sebanyak yang diinginkan, sebaiknya minimall 100. hasil pengambilan gambar akan disimpan di dalam folder `dataset`
2. buatlah folder `grayscale` di bawah folder dataset, penamaan folder sesuai dengan nama Anda karena nanti akan digunakan sebagai label ketika testing.
3. jalankan file `data-preparation.py` untuk membuat bahan untuk training yang akan datang. pada langkah ini akan terbentuk file *.pickle pada direktori saat ini.
4. buka file `face-training.py` kemudian ganti nilai pada variabel `class_number` menjadi angka seusai dengan jumlah folder yang berada di bawah folder `grayscale`. variabel tersebut digunakan sebagai indikasi jumlah output atau jumlah klasifikasi, sehingga jika nanti melakukan pengambilan gambar dengan label yang berbeda maka perlu dilakukan perubahan pada variabel `class_number`
5. yang terakhir adalah menjalankan file `face-testing.py`, jika langkah 1-4 secara benar dilakukan seharusnya nanti akan muncul windows baru seperti pada langkah 1 tetapi pada tahap ini akan mengenali objek 