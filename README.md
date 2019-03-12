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