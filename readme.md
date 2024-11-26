# **Prediksi Pendapatan Tahunan Pelanggan**

Proyek ini menggunakan **TensorFlow.js** untuk membangun model **Regresi Linear Multivariat** yang memprediksi pendapatan tahunan pelanggan berdasarkan usia, jenis kelamin, pendidikan, pengeluaran bulanan, jumlah anggota keluarga, dan lokasi tempat tinggal.

---

## **Cara Instalasi**

1. **Persiapan Proyek**
   Pastikan Node.js telah terinstall di sistem Anda. Jika belum, download dan install dari [Node.js](https://nodejs.org).

2. **Clone Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

3. **Install Dependencies**
   ```bash
   npm install @tensorflow/tfjs
   ```

4. **Jalankan Proyek**
   ```bash
   node index.js
   ```

---

## **Algoritma yang Digunakan**

### **1. Regresi Linear Multivariat**
Algoritma regresi digunakan untuk memodelkan hubungan antara beberapa variabel independen (fitur) dan variabel dependen (target).

- **Input Fitur**:
  - Usia
  - Jenis Kelamin
  - Pendidikan
  - Pengeluaran Bulanan
  - Jumlah Anggota Keluarga
  - Lokasi
- **Target Output**:
  - Pendapatan Tahunan (dalam juta Rupiah)

Model dibangun dengan menggunakan lapisan-lapisan dense (**fully connected layers**) untuk memprediksi target.

---

## **Penjelasan Baris demi Baris Kode**

### **1. Import TensorFlow.js**
```javascript
const tf = require('@tensorflow/tfjs');
```
- **Fungsi**: Mengimpor pustaka TensorFlow.js untuk membangun model ML di Node.js.

---

### **2. Dataset**
```javascript
const data = [
  { usia: 25, jenisKelamin: 1, pendidikan: 4, pengeluaran: 5, keluarga: 3, lokasi: 1, pendapatan: 120 },
  { usia: 40, jenisKelamin: 0, pendidikan: 5, pengeluaran: 10, keluarga: 4, lokasi: 1, pendapatan: 300 },
  { usia: 35, jenisKelamin: 1, pendidikan: 3, pengeluaran: 6, keluarga: 5, lokasi: 2, pendapatan: 200 },
  { usia: 29, jenisKelamin: 0, pendidikan: 2, pengeluaran: 3, keluarga: 2, lokasi: 3, pendapatan: 80 },
  { usia: 50, jenisKelamin: 1, pendidikan: 6, pengeluaran: 20, keluarga: 2, lokasi: 1, pendapatan: 500 },
];
```
- **Fungsi**: Dataset mentah yang berisi fitur input dan target output.
- Kolom:
  - `usia`: Usia pelanggan (dalam tahun).
  - `jenisKelamin`: 1 untuk pria, 0 untuk wanita.
  - `pendidikan`: Skala ordinal (1=SD hingga 6=Doktor).
  - `pengeluaran`: Pengeluaran bulanan (dalam juta Rupiah).
  - `keluarga`: Jumlah anggota keluarga.
  - `lokasi`: Lokasi tempat tinggal (1=Perkotaan, 2=Pinggiran, 3=Pedesaan).
  - `pendapatan`: Pendapatan tahunan (dalam juta Rupiah).

---

### **3. Membentuk Tensor Input dan Output**
```javascript
const xs = tf.tensor2d(
  data.map((item) => [
    item.usia,
    item.jenisKelamin,
    item.pendidikan,
    item.pengeluaran,
    item.keluarga,
    item.lokasi,
  ])
);
const ys = tf.tensor2d(data.map((item) => [item.pendapatan]));
```
- **Fungsi**: Membentuk tensor dari data mentah.
  - `xs`: Tensor untuk fitur input (usia, jenis kelamin, dll.).
  - `ys`: Tensor untuk target output (pendapatan).

---

### **4. Membuat Model**
```javascript
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, inputShape: [6], activation: 'relu' })); // Hidden Layer
model.add(tf.layers.dense({ units: 1 })); // Output Layer
model.compile({
  optimizer: tf.train.adam(),
  loss: 'meanSquaredError',
});
```
- **Fungsi**: Membuat model regresi menggunakan lapisan dense.
  - `units: 10`: Jumlah neuron di hidden layer pertama.
  - `inputShape: [6]`: Model menerima 6 fitur input.
  - `activation: 'relu'`: Fungsi aktivasi Rectified Linear Unit.
  - `optimizer: 'adam'`: Algoritma optimasi untuk mempercepat konvergensi.
  - `loss: 'meanSquaredError'`: Fungsi loss untuk regresi.

---

### **5. Melatih Model**
```javascript
await model.fit(xs, ys, { epochs: 200 });
console.log('Model telah dilatih.');
```
- **Fungsi**: Melatih model menggunakan data input (`xs`) dan target (`ys`).
  - `epochs: 200`: Model akan melalui dataset sebanyak 200 kali.

---

### **6. Prediksi Data Baru**
```javascript
const input = tf.tensor2d([[30, 1, 4, 7, 3, 1]]); // Input data mentah
const prediction = model.predict(input);
prediction.print();
```
- **Fungsi**: Membuat prediksi menggunakan data baru.
  - Contoh input: `[[30, 1, 4, 7, 3, 1]]`
    - Usia: 30 tahun
    - Jenis Kelamin: Pria
    - Pendidikan: Sarjana
    - Pengeluaran Bulanan: 7 juta
    - Jumlah Anggota Keluarga: 3
    - Lokasi: Perkotaan
- **Output**: Prediksi pendapatan tahunan pelanggan.

---

### **7. Menampilkan Hasil**
```javascript
console.log(`Pendapatan Tahunan yang Diprediksi: ${prediction.dataSync()[0]} juta`);
```
- **Fungsi**: Menampilkan hasil prediksi langsung dalam skala asli (dalam juta Rupiah).

---

## **Catatan**
1. Dataset harus cukup besar untuk meningkatkan akurasi prediksi.
2. Model ini dapat diperluas dengan data tambahan seperti pekerjaan, pengalaman kerja, atau pendapatan keluarga.

---